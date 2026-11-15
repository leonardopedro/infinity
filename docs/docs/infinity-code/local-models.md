---
sidebar_position: 5
title: Local Models
---

# Local Models

Infinity Code does not call a cloud API directly; it talks to model providers
(see [Model Providers](./model-providers.md)). The
`infinity-provider-openai` provider speaks the OpenAI Chat Completions API, so any local inference server
that implements it can back the agent: Edge0's `edge0-infer serve`, Ollama,
vLLM, llama.cpp's server, LM Studio, or a self-hosted compatible endpoint.

This page walks through **Edge0 35B** (`edge0-infer`, the native GGML engine),
first on the same machine, then inside an [OpenShell](https://docs.nvidia.com/openshell/latest/index.html) sandbox.

## Serve Edge0 35B

Build `edge0-infer` and convert the model assets as described in the
[Edge0 getting started guide](https://github.com/Edge0-AI/Edge0/blob/main/edge0-infer/docs/getting-started.md).
Upstream documents the CPU preset on Linux; a Vulkan build works too (configure
with `GGML_VULKAN=ON`) and is what the numbers below come from:

```bash
# CPU (upstream-supported path)
cmake --preset cpu && cmake --build --preset cpu
build/cpu/bin/edge0-infer serve --model "$MOE_ASSETS_35B" --device cpu

# Vulkan on Linux: ~2.4× decode and ~4× prefill of the CPU path on a GTX 1060
cmake --preset linux-vulkan && cmake --build --preset linux-vulkan
GGML_VK_VISIBLE_DEVICES=1 build/linux-vulkan/bin/edge0-infer serve \
  --model "$MOE_ASSETS_35B" --device vulkan --model-option expert_slots=24
```

`GGML_VK_VISIBLE_DEVICES=1` is required on a machine with a second Vulkan
device: without it the engine picks the first one (an Intel iGPU, for example)
and dies with `vk::DeviceLostError`. `expert_slots=24` is the sweet spot for a
6 GiB card (8/16/32 are the other sensible values): it keeps the full 16 384-token
context under ~5 GiB of VRAM, and more slots trade memory headroom for a few
percent of decode speed.

The server listens on `http://127.0.0.1:8000/v1` and answers to the model alias
from the asset `model.json` (`edge0_35b`). Its Chat Completions endpoint supports
streaming and function calling, which is everything Infinity needs.

:::note
`edge0-infer` runs one inference at a time and answers `429 busy` while a
request is in flight. The provider classifies that as `Throttled`, so the
runtime backs off and retries instead of failing the turn.
:::

## Install the provider

```bash
infinity provider install edge0 --crate infinity-provider-openai \
  --path crates/infinity-provider-openai
```

This `cargo install`s the binary and registers it in
`~/.infinity/providers.json`. Edit the entry to pass the endpoint and model:

```json
{
  "edge0": {
    "command": [
      "infinity-provider-openai",
      "--base-url", "http://127.0.0.1:8000/v1",
      "--model", "edge0_35b",
      "--context-window", "16384"
    ]
  }
}
```

`--context-window 16384` is the 35B's recommended session capacity; it drives
compaction. Leave `--max-output-tokens` unset to let the engine apply its own
default and trim it to the remaining context.

Restart the daemon (`infinity daemon restart`) so it re-reads the file, then pick
the model with `/model` in the TUI. Providers are registered in file order, and
the first model of the first provider is the default for new sessions, so put
`edge0` first to make it the default.

### Provider flags

| Flag | Environment variable | Default |
| --- | --- | --- |
| `--base-url` | `INFINITY_OPENAI_BASE_URL` | `http://127.0.0.1:8000/v1` |
| `--api-key` | `INFINITY_OPENAI_API_KEY` | none |
| `--model` (repeatable, comma-separated) | `INFINITY_OPENAI_MODELS` | discovered from `GET /models` |
| `--display-name` | `INFINITY_OPENAI_DISPLAY_NAME` | the model id |
| `--context-window` | `INFINITY_OPENAI_CONTEXT_WINDOW` | `32768` |
| `--max-output-tokens` | `INFINITY_OPENAI_MAX_OUTPUT_TOKENS` | none (server default) |
| `--parallel-tool-calls` | `INFINITY_OPENAI_PARALLEL_TOOL_CALLS` | `false` |

Prefer the environment variables for a token: the provider process inherits the
daemon's environment, and `providers.json` is world-readable.

## Run it inside OpenShell

OpenShell sandboxes reach host-local services through
`host.openshell.internal`, never `127.0.0.1`. Two things follow: the model server
must listen on a non-loopback address, and the sandbox needs a provider profile
that authorizes that endpoint.

### 1. Serve Edge0 for the sandbox

A non-loopback listener requires `--api-key-file`, and clients must send its
token:

```bash
install -d -m 0700 ~/.edge0
head -c 32 /dev/urandom | base64 | tr -d '\n' > ~/.edge0/api-key
chmod 600 ~/.edge0/api-key

build/cpu/bin/edge0-infer serve --model "$MOE_ASSETS_35B" --device cpu \
  --host 0.0.0.0 --port 8000 --api-key-file ~/.edge0/api-key
```

For the GPU (Vulkan) build, serve on **8001** instead and pick the `edge0-gpu`
profile in step 2, so a GPU and a CPU server can coexist and a sandbox chooses
one by profile name:

```bash
GGML_VK_VISIBLE_DEVICES=1 build/linux-vulkan/bin/edge0-infer serve \
  --model "$MOE_ASSETS_35B" --device vulkan --model-option expert_slots=24 \
  --host 0.0.0.0 --port 8001 --api-key-file ~/.edge0/api-key
```

### 2. Import the provider profile

The `edge0-openai` profile in the OpenShell repository declares the endpoint
(`host.openshell.internal:8000`) and the binaries allowed to call it. Import it
and create a provider instance:

```bash
openshell provider profile lint   -f providers/edge0-openai.yaml
openshell provider profile import -f providers/edge0-openai.yaml
openshell provider create --name edge0 --type edge0-openai
```

For the GPU endpoint, import
[`providers/edge0-openai-gpu.yaml`](https://github.com/NVIDIA/OpenShell/blob/main/providers/edge0-openai-gpu.yaml)
instead — same binaries and credential model, port 8001 — and name the instance
`edge0-gpu`:

```bash
openshell provider profile lint   -f providers/edge0-openai-gpu.yaml
openshell provider profile import -f providers/edge0-openai-gpu.yaml
openshell provider create --name edge0-gpu --type edge0-gpu
```

The profile is credentialless: the Edge0 token is an ordinary environment
variable inside the sandbox, so OpenShell's credential proxy is not involved.
The profile allows the provider binary at `/nix/store/*-infinity/bin/`, which is
what the launcher below produces, plus the usual install locations; edit
`binaries` in your copy if the binaries live somewhere else, or every request
is denied.

### 3. Bake Infinity in through the Nix store

The OpenShell fork's VM driver mounts the host's `/nix` read-only into the guest
(VirtioFS + DAX), so a store path on the host is the same path in the guest.
`openshell-nixos/infinity-sandbox` uses that instead of `sandbox upload`:

```bash
cargo build --release -p infinity-agent-cli -p infinity-provider-openai -p sandbox-local

PROJECTS_DIR=~/code ../OpenShell/openshell-nixos/infinity-sandbox

# GPU model: selects the edge0-gpu profile and host.openshell.internal:8001
SANDBOX_GPU=1 PROJECTS_DIR=~/code ../OpenShell/openshell-nixos/infinity-sandbox
```

It adds the three binaries to the store with `nix store add-path`, puts
`ripgrep`/`git` from nixpkgs on the guest's `PATH` (also through `/nix`), stages
`~/.infinity` into `$PROJECTS_DIR/.infinity-home` as the guest's `HOME`, and
creates the sandbox on the **cloud-hypervisor** backend with two mounts: `/nix`
read-only and the project directory read-write at `/workspace`. The generated
config points the provider at `http://host.openshell.internal:8000/v1` and
registers `sandbox-local --enable-sandbox=false`, since OpenShell's VM is
already the isolation boundary.

Upgrading Infinity is rebuilding and rerunning the launcher — the new binaries
become a new store path; nothing is uploaded and no image is rebuilt. The
launcher needs a gateway whose driver has a cloud-hypervisor VMM and guest
kernel configured (`OPENSHELL_VM_CH_PATH`, `OPENSHELL_VM_CH_KERNEL`); see
[`openshell-nixos/infinity-sandbox.md`](https://github.com/NVIDIA/OpenShell/blob/main/openshell-nixos/infinity-sandbox.md)
for the prerequisites, the kernel download and the backend setup, plus the
manual `openshell sandbox create` command (`DRY_RUN=1` prints it).

### Verify the wiring

Check each hop separately before blaming the agent:

```bash
# 1. The host serves the model and accepts the token
curl -sS http://host.openshell.internal:8000/v1/models \
  -H "Authorization: Bearer $INFINITY_OPENAI_API_KEY"

# 2. The profile granted the sandbox access
openshell policy get infinity --full   # from the host (the launcher's default name)

# 3. A completion round-trips from inside the sandbox
curl -sS http://host.openshell.internal:8000/v1/chat/completions \
  -H "Authorization: Bearer $INFINITY_OPENAI_API_KEY" \
  -H 'content-type: application/json' \
  -d '{"model":"edge0_35b","messages":[{"role":"user","content":"ping"}],"max_tokens":16}'
```

A denied request usually means the caller binary is not listed in the profile's
`binaries`, or the profile is not attached
(`openshell sandbox provider list infinity`). Launch a new process after
attaching a provider — running processes do not receive new environment
variables.

## Limitations

- **No image input.** The provider reports `supports_image_input: false`, so the
  runtime replaces image tool results with a text placeholder. `edge0-infer`
  rejects multimodal content anyway.
- **One request at a time.** Edge0 serializes inference; parallel Infinity
  threads wait through `429 busy` retries. Use a server that queues (or several
  endpoints) if that matters.
- **Linux Vulkan is not upstream-verified.** The engine's tested Vulkan path is
  Windows; Linux builds and runs, but Pascal cards have no cooperative-matrix
  fast paths, so expect the numbers above rather than the Windows figures.
  Fall back to `--device cpu` if the device is lost.
- **No prompt caching hints.** `edge0-infer` reuses the conversation prefix
  automatically; no cache-control fields are sent.
- **Tool calls execute one at a time.** The provider asks for
  `parallel_tool_calls: false` because Infinity executes only the first call of
  a turn.
