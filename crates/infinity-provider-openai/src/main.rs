//! Standalone binary serving the OpenAI-compatible [`ModelProvider`] over a
//! Unix socket (see `infinity_provider_protocol::remote`).
//!
//! Prints the generated socket path as the first stdout line so the
//! supervising daemon can discover it; logs go to stderr.
//!
//! ```sh
//! # Edge0 35B served by `edge0-infer serve` on its default loopback address
//! infinity-provider-openai --model edge0_35b --context-window 16384
//! ```

use std::io::Write;
use std::sync::Arc;

use clap::Parser;
use infinity_provider_openai::{OpenAiCompatibleConfig, OpenAiCompatibleProvider};
use infinity_provider_protocol::remote::serve_provider;
use tracing_subscriber::EnvFilter;

type BoxError = Box<dyn std::error::Error + Send + Sync>;

#[derive(Parser, Debug)]
#[command(
    about = "OpenAI Chat Completions compatible model provider for Infinity",
    long_about = None
)]
struct Args {
    /// Base URL of the OpenAI-compatible server, including the API version.
    #[arg(
        long,
        env = "INFINITY_OPENAI_BASE_URL",
        default_value = infinity_provider_openai::DEFAULT_BASE_URL
    )]
    base_url: String,

    /// Bearer token for the server. Local servers accept any non-empty value.
    #[arg(long, env = "INFINITY_OPENAI_API_KEY")]
    api_key: Option<String>,

    /// Model id(s) to advertise; repeat or comma-separate. Empty discovers
    /// them from `GET /models`.
    #[arg(long = "model", env = "INFINITY_OPENAI_MODELS", value_delimiter = ',')]
    models: Vec<String>,

    /// Display name shown in the model picker.
    #[arg(long, env = "INFINITY_OPENAI_DISPLAY_NAME")]
    display_name: Option<String>,

    /// Context window reported for every model (compaction threshold).
    #[arg(long, env = "INFINITY_OPENAI_CONTEXT_WINDOW", default_value_t = 32768)]
    context_window: usize,

    /// Maximum output tokens reported for every model; omitted lets the server
    /// use its own default.
    #[arg(long, env = "INFINITY_OPENAI_MAX_OUTPUT_TOKENS")]
    max_output_tokens: Option<u64>,

    /// Ask the server for parallel tool calls. Infinity executes only the
    /// first call of a turn, so this is off by default.
    #[arg(
        long,
        env = "INFINITY_OPENAI_PARALLEL_TOOL_CALLS",
        default_value_t = false
    )]
    parallel_tool_calls: bool,
}

#[tokio::main]
async fn main() -> Result<(), BoxError> {
    // stdout is reserved for the socket path; log to stderr (captured into
    // the supervising daemon's log, so no ANSI styling).
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .with_writer(std::io::stderr)
        .with_ansi(false)
        .init();

    let args = Args::parse();
    let provider = Arc::new(OpenAiCompatibleProvider::new(OpenAiCompatibleConfig {
        base_url: args.base_url.clone(),
        api_key: args.api_key,
        models: args.models,
        display_name: args.display_name,
        context_window: args.context_window,
        max_output_tokens: args.max_output_tokens,
        parallel_tool_calls: args.parallel_tool_calls,
    })?);

    tracing::info!(base_url = %args.base_url, "serving OpenAI-compatible provider");

    let (socket_path, server) = serve_provider(provider)?;

    println!("{}", socket_path.display());
    std::io::stdout().flush()?;

    server.await;
    Ok(())
}
