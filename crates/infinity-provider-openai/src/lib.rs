//! OpenAI Chat Completions compatible model provider for Infinity.
//!
//! Infinity's provider protocol is deliberately backend-agnostic; this crate
//! implements it against any server that speaks the OpenAI Chat Completions
//! API with streaming and function calling. That covers local inference
//! servers such as Edge0's `edge0-infer serve` (the 35B native engine), as
//! well as Ollama, vLLM, llama.cpp's server, LM Studio and hosted
//! OpenAI-compatible endpoints.
//!
//! The provider is configured from flags (or the matching environment
//! variables, see [`OpenAiCompatibleConfig`]) and exposes either the models
//! named on the command line or whatever `GET /models` reports.
//!
//! ```no_run
//! # async fn example() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
//! use infinity_provider_openai::{OpenAiCompatibleConfig, OpenAiCompatibleProvider};
//!
//! let config = OpenAiCompatibleConfig {
//!     base_url: "http://127.0.0.1:8000/v1".to_owned(),
//!     models: vec!["edge0_35b".to_owned()],
//!     context_window: 16384,
//!     ..Default::default()
//! };
//! let provider = OpenAiCompatibleProvider::new(config)?;
//! # let _ = provider;
//! # Ok(())
//! # }
//! ```

mod convert;
mod stream;

use std::time::Duration;

use async_trait::async_trait;
use infinity_provider_protocol::{
    CompletionError, CompletionRequest, ErrorClass, ModelEntry, ModelProvider, ModelStream,
};
use serde_json::Value;

pub use convert::build_chat_request;
pub use stream::parse_sse_stream;

type BoxError = Box<dyn std::error::Error + Send + Sync>;

/// Default base URL of a local OpenAI-compatible server (`edge0-infer serve`
/// listens here with no arguments).
pub const DEFAULT_BASE_URL: &str = "http://127.0.0.1:8000/v1";

/// How long to wait for the TCP connection to the inference server. There is
/// deliberately no overall request timeout: on CPU-only hardware a large
/// prefill can take minutes before the first token, and the runtime applies
/// no timeouts of its own.
const CONNECT_TIMEOUT: Duration = Duration::from_secs(30);

/// Configuration for an OpenAI Chat Completions compatible endpoint.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OpenAiCompatibleConfig {
    /// Base URL including the API version, e.g. `http://127.0.0.1:8000/v1`.
    pub base_url: String,
    /// Bearer token sent as `Authorization`. Local servers ignore it, but many
    /// clients (and `edge0-infer serve` on a non-loopback address) expect one.
    pub api_key: Option<String>,
    /// Models to advertise, in order (the first is the default). Empty means
    /// discover them from `GET /models`.
    pub models: Vec<String>,
    /// Display name used for every advertised model. Defaults to the model id.
    pub display_name: Option<String>,
    /// Context window reported for every advertised model (drives compaction).
    pub context_window: usize,
    /// Maximum output tokens reported for every advertised model; `None` lets
    /// the server apply its own default.
    pub max_output_tokens: Option<u64>,
    /// Ask the server for parallel tool calls. Infinity executes only the
    /// first call of a turn, so this is off by default.
    pub parallel_tool_calls: bool,
}

impl Default for OpenAiCompatibleConfig {
    fn default() -> Self {
        Self {
            base_url: DEFAULT_BASE_URL.to_owned(),
            api_key: None,
            models: Vec::new(),
            display_name: None,
            context_window: 32768,
            max_output_tokens: None,
            parallel_tool_calls: false,
        }
    }
}

impl OpenAiCompatibleConfig {
    fn chat_completions_url(&self) -> String {
        format!("{}/chat/completions", self.base_url.trim_end_matches('/'))
    }

    fn models_url(&self) -> String {
        format!("{}/models", self.base_url.trim_end_matches('/'))
    }
}

/// A [`ModelProvider`] backed by an OpenAI Chat Completions endpoint.
pub struct OpenAiCompatibleProvider {
    client: reqwest::Client,
    config: OpenAiCompatibleConfig,
}

impl OpenAiCompatibleProvider {
    pub fn new(config: OpenAiCompatibleConfig) -> Result<Self, BoxError> {
        let client = reqwest::Client::builder()
            .connect_timeout(CONNECT_TIMEOUT)
            .build()?;
        Ok(Self { client, config })
    }

    fn authorize(&self, request: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        match &self.config.api_key {
            Some(key) => request.bearer_auth(key),
            None => request,
        }
    }

    fn entries_for(&self, model_ids: Vec<String>) -> Vec<ModelEntry> {
        model_ids
            .into_iter()
            .map(|model_id| ModelEntry {
                display_name: self
                    .config
                    .display_name
                    .clone()
                    .unwrap_or_else(|| model_id.clone()),
                model_id,
                context_window: self.config.context_window,
                max_output_tokens: self.config.max_output_tokens,
                supports_image_input: false,
            })
            .collect()
    }
}

#[async_trait]
impl ModelProvider for OpenAiCompatibleProvider {
    async fn list_models(&self) -> Result<Vec<ModelEntry>, BoxError> {
        if !self.config.models.is_empty() {
            return Ok(self.entries_for(self.config.models.clone()));
        }

        let response = self
            .authorize(self.client.get(self.config.models_url()))
            .send()
            .await
            .map_err(|e| format!("failed to reach {}: {e}", self.config.models_url()))?;
        let status = response.status();
        let body = response.text().await?;
        if !status.is_success() {
            return Err(format!("{} returned {status}: {body}", self.config.models_url()).into());
        }

        let value: Value = serde_json::from_str(&body)
            .map_err(|e| format!("{} returned invalid JSON: {e}", self.config.models_url()))?;
        let ids = value
            .get("data")
            .and_then(Value::as_array)
            .ok_or("model list response has no `data` array")?
            .iter()
            .filter_map(|model| model.get("id").and_then(Value::as_str))
            .map(str::to_owned)
            .collect::<Vec<_>>();
        if ids.is_empty() {
            return Err("the server advertised no models; pass --model explicitly".into());
        }
        Ok(self.entries_for(ids))
    }

    async fn invoke_model(
        &self,
        model_id: &str,
        request: CompletionRequest,
    ) -> Result<ModelStream, CompletionError> {
        let body = build_chat_request(model_id, &request, self.config.parallel_tool_calls)?;

        let response = self
            .authorize(self.client.post(self.config.chat_completions_url()))
            .header(reqwest::header::ACCEPT, "text/event-stream")
            .json(&body)
            .send()
            .await
            .map_err(|e| {
                CompletionError::provider(
                    ErrorClass::Transient,
                    format!("failed to reach the inference server: {e}"),
                )
            })?;

        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            return Err(stream::error_from_response(status, &body));
        }

        Ok(parse_sse_stream(response))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_points_at_a_local_server() {
        let config = OpenAiCompatibleConfig::default();
        assert_eq!(config.base_url, DEFAULT_BASE_URL);
        assert_eq!(
            config.chat_completions_url(),
            "http://127.0.0.1:8000/v1/chat/completions"
        );
        assert_eq!(
            OpenAiCompatibleConfig {
                base_url: "http://host:1/v1/".to_owned(),
                ..Default::default()
            }
            .models_url(),
            "http://host:1/v1/models"
        );
    }

    #[test]
    fn entries_use_the_configured_metadata() {
        let provider = OpenAiCompatibleProvider::new(OpenAiCompatibleConfig {
            models: vec!["edge0_35b".to_owned()],
            display_name: Some("Edge0 35B".to_owned()),
            context_window: 16384,
            max_output_tokens: Some(8192),
            ..Default::default()
        })
        .expect("build provider");
        let entries = provider.entries_for(provider.config.models.clone());
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].model_id, "edge0_35b");
        assert_eq!(entries[0].display_name, "Edge0 35B");
        assert_eq!(entries[0].context_window, 16384);
        assert_eq!(entries[0].max_output_tokens, Some(8192));
        assert!(!entries[0].supports_image_input);
    }
}
