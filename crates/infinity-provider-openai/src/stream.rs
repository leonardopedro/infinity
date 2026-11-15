//! Server-sent-events parsing for OpenAI Chat Completions streams, and error
//! classification for non-2xx responses.

use std::collections::BTreeMap;

use futures_util::StreamExt;
use infinity_provider_protocol::{
    CompletionError, ErrorClass, FinalResponse, ModelStream, StreamChunk, ToolCall,
    ToolCallDeltaContent, Usage,
};
use reqwest::StatusCode;
use serde::Deserialize;
use serde_json::Value;

// ── Error classification ──

/// Turn a non-2xx response from the inference server into a classified
/// error. The classification drives the runtime's retry behavior, so it is
/// worth getting right: `edge0-infer serve` answers `429 busy` while one
/// inference is running (a retry after backoff succeeds) and
/// `400 context_length_exceeded` when the prompt does not fit (the runtime
/// must shrink the input instead of retrying).
pub fn error_from_response(status: StatusCode, body: &str) -> CompletionError {
    let value: Value = serde_json::from_str(body).unwrap_or(Value::Null);
    error_from_value(status.as_u16(), &value, body)
}

fn error_from_value(status: u16, value: &Value, raw_body: &str) -> CompletionError {
    let code = value
        .pointer("/error/code")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let kind = value
        .pointer("/error/type")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let message = value
        .pointer("/error/message")
        .and_then(Value::as_str)
        .map(str::to_owned)
        .unwrap_or_else(|| raw_body.trim().to_owned());
    let message = if message.is_empty() {
        "no error details".to_owned()
    } else {
        message
    };
    let class = classify(status, code, kind, &message);
    CompletionError::provider(
        class,
        format!("inference server returned {status}: {message}"),
    )
}

fn classify(status: u16, code: &str, kind: &str, message: &str) -> ErrorClass {
    let code = code.to_ascii_lowercase();
    let kind = kind.to_ascii_lowercase();
    let message = message.to_ascii_lowercase();

    let mentions_context = code.contains("context_length")
        || code.contains("context_capacity")
        || code.contains("request_too_large")
        || kind.contains("context")
        || message.contains("context length")
        || message.contains("context window")
        || message.contains("maximum context")
        || message.contains("capacity");

    // One inference at a time, or a rate limit: wait and retry. Servers also
    // report these inside a stream that already started, where the HTTP status
    // is 200, so the code and kind matter as much as the status.
    let mentions_busy = code.contains("busy")
        || code.contains("rate_limit")
        || code.contains("overloaded")
        || kind.contains("rate_limit")
        || kind.contains("overloaded");

    match status {
        _ if mentions_context => ErrorClass::ContextOverflow,
        _ if mentions_busy || status == 429 => ErrorClass::Throttled,
        408 => ErrorClass::Transient,
        413 => ErrorClass::ContextOverflow,
        // Backend hiccups and shutdowns: a retry may succeed.
        _ if (500..=599).contains(&status) => ErrorClass::Transient,
        _ => ErrorClass::Fatal,
    }
}

// ── Stream chunks ──

#[derive(Debug, Default, Deserialize)]
struct ChunkPayload {
    #[serde(default)]
    choices: Vec<ChoicePayload>,
    usage: Option<UsagePayload>,
}

#[derive(Debug, Default, Deserialize)]
struct ChoicePayload {
    #[serde(default)]
    delta: DeltaPayload,
    #[serde(default)]
    finish_reason: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
struct DeltaPayload {
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    reasoning_content: Option<String>,
    #[serde(default)]
    tool_calls: Vec<ToolCallPayload>,
}

#[derive(Debug, Deserialize)]
struct ToolCallPayload {
    #[serde(default)]
    index: usize,
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    function: Option<FunctionPayload>,
}

#[derive(Debug, Default, Deserialize)]
struct FunctionPayload {
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    arguments: Option<String>,
}

#[derive(Debug, Deserialize)]
struct UsagePayload {
    #[serde(default)]
    prompt_tokens: u64,
    #[serde(default)]
    completion_tokens: u64,
    #[serde(default)]
    total_tokens: u64,
    #[serde(default)]
    prompt_tokens_details: Option<PromptTokensDetails>,
}

#[derive(Debug, Deserialize)]
struct PromptTokensDetails {
    #[serde(default)]
    cached_tokens: Option<u64>,
}

impl UsagePayload {
    fn into_usage(self) -> Usage {
        Usage {
            input_tokens: self.prompt_tokens,
            output_tokens: self.completion_tokens,
            total_tokens: if self.total_tokens == 0 {
                self.prompt_tokens + self.completion_tokens
            } else {
                self.total_tokens
            },
            cached_input_tokens: self
                .prompt_tokens_details
                .and_then(|details| details.cached_tokens)
                .unwrap_or(0),
        }
    }
}

/// A tool call being accumulated from argument deltas.
#[derive(Debug, Default)]
struct PartialCall {
    id: String,
    name: String,
    arguments: String,
    name_emitted: bool,
}

/// Adapt a streaming `POST /chat/completions` response into a [`ModelStream`].
///
/// Text and `reasoning_content` deltas are forwarded as they arrive. Tool
/// calls are accumulated per index, because a server may interleave the
/// argument fragments of parallel calls, and emitted as complete
/// [`StreamChunk::ToolCall`]s at `finish_reason` (or the end of the stream),
/// which is what the runtime executes; the argument fragments are also forwarded as
/// [`StreamChunk::ToolCallDelta`]s for display. A final chunk carrying usage is
/// always emitted, even if the connection ends without `[DONE]`.
pub fn parse_sse_stream(response: reqwest::Response) -> ModelStream {
    Box::pin(async_stream::stream! {
        let mut buffer: Vec<u8> = Vec::new();
        let mut body = response.bytes_stream();
        let mut calls: BTreeMap<usize, PartialCall> = BTreeMap::new();
        let mut usage: Option<Usage> = None;

        while let Some(chunk) = body.next().await {
            let chunk = match chunk {
                Ok(chunk) => chunk,
                Err(error) => {
                    yield Err(CompletionError::provider(
                        ErrorClass::Transient,
                        format!("response stream failed: {error}"),
                    ));
                    return;
                }
            };
            buffer.extend_from_slice(&chunk);

            while let Some(newline) = buffer.iter().position(|byte| *byte == b'\n') {
                let line: Vec<u8> = buffer.drain(..=newline).collect();
                let line = String::from_utf8_lossy(&line);
                let line = line.trim_end_matches(['\n', '\r']);
                if line.is_empty() || line.starts_with(':') {
                    continue;
                }
                let Some(data) = line.strip_prefix("data:") else {
                    continue;
                };
                let data = data.trim();
                if data == "[DONE]" {
                    for chunk in finish_calls(&mut calls) {
                        yield chunk;
                    }
                    yield Ok(StreamChunk::Final(FinalResponse { usage: usage.take() }));
                    return;
                }

                match handle_event(data, &mut calls, &mut usage) {
                    Ok(chunks) => {
                        for chunk in chunks {
                            yield chunk;
                        }
                    }
                    Err(error) => {
                        yield Err(error);
                        return;
                    }
                }
            }
        }

        // The server closed the stream without `[DONE]`; flush whatever is
        // pending so the turn still terminates.
        for chunk in finish_calls(&mut calls) {
            yield chunk;
        }
        yield Ok(StreamChunk::Final(FinalResponse { usage }));
    })
}

fn handle_event(
    data: &str,
    calls: &mut BTreeMap<usize, PartialCall>,
    usage: &mut Option<Usage>,
) -> Result<Vec<Result<StreamChunk, CompletionError>>, CompletionError> {
    let value: Value = serde_json::from_str(data).map_err(|error| {
        CompletionError::ResponseError(format!("invalid stream event: {error}: {data}"))
    })?;

    if value.get("error").is_some() {
        // Errors after the stream started arrive as an event, not a status.
        return Err(error_from_value(200, &value, ""));
    }

    let payload: ChunkPayload = serde_json::from_value(value).map_err(|error| {
        CompletionError::ResponseError(format!("invalid stream chunk: {error}"))
    })?;

    let mut chunks = Vec::new();
    if let Some(reported) = payload.usage {
        *usage = Some(reported.into_usage());
    }

    for choice in payload.choices {
        let delta = choice.delta;
        if let Some(text) = delta.content.filter(|text| !text.is_empty()) {
            chunks.push(Ok(StreamChunk::Text(text)));
        }
        if let Some(text) = delta.reasoning_content.filter(|text| !text.is_empty()) {
            chunks.push(Ok(StreamChunk::ReasoningDelta { id: None, text }));
        }

        for (position, tool_call) in delta.tool_calls.into_iter().enumerate() {
            let index = tool_call.index;
            let entry = calls.entry(index).or_insert_with(|| PartialCall {
                id: tool_call
                    .id
                    .clone()
                    .unwrap_or_else(|| format!("call_{index}_{position}")),
                ..Default::default()
            });
            if let Some(id) = tool_call.id {
                // Later chunks omit the id; the first one carries it.
                entry.id = id;
            }
            if let Some(function) = tool_call.function {
                if let Some(name) = function.name.filter(|name| !name.is_empty())
                    && !entry.name_emitted
                {
                    entry.name_emitted = true;
                    entry.name = name.clone();
                    chunks.push(Ok(StreamChunk::ToolCallDelta {
                        id: entry.id.clone(),
                        content: ToolCallDeltaContent::Name(name),
                    }));
                }
                if let Some(fragment) = function.arguments.filter(|fragment| !fragment.is_empty()) {
                    entry.arguments.push_str(&fragment);
                    chunks.push(Ok(StreamChunk::ToolCallDelta {
                        id: entry.id.clone(),
                        content: ToolCallDeltaContent::Delta(fragment),
                    }));
                }
            }
        }

        if choice.finish_reason.is_some() {
            chunks.extend(finish_calls(calls));
        }
    }

    Ok(chunks)
}

fn finish_calls(
    calls: &mut BTreeMap<usize, PartialCall>,
) -> Vec<Result<StreamChunk, CompletionError>> {
    std::mem::take(calls)
        .into_values()
        .flat_map(complete_call)
        .collect()
}

fn complete_call(call: PartialCall) -> Option<Result<StreamChunk, CompletionError>> {
    if call.name.is_empty() {
        // A call that never named a function is not executable; the server
        // already reported it as rejected.
        return None;
    }
    let arguments = if call.arguments.trim().is_empty() {
        serde_json::json!({})
    } else {
        match serde_json::from_str(&call.arguments) {
            Ok(arguments) => arguments,
            Err(error) => {
                return Some(Err(CompletionError::provider(
                    ErrorClass::Transient,
                    format!(
                        "streamed arguments for tool {} were not valid JSON: {error}",
                        call.name
                    ),
                )));
            }
        }
    };
    Some(Ok(StreamChunk::ToolCall(ToolCall::new(
        call.id, call.name, arguments,
    ))))
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::Router;
    use axum::routing::post;
    use futures_util::StreamExt;
    use infinity_provider_protocol::ModelProvider as _;
    use tokio::net::TcpListener;

    async fn serve_sse(body: &'static str) -> String {
        let app = Router::new().route(
            "/v1/chat/completions",
            post(move || async move {
                (
                    [
                        ("content-type", "text/event-stream"),
                        ("cache-control", "no-cache"),
                    ],
                    body,
                )
            }),
        );
        let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
        let address = listener.local_addr().expect("address");
        tokio::spawn(async move {
            axum::serve(listener, app).await.expect("serve");
        });
        format!("http://{address}/v1")
    }

    async fn collect(base_url: String) -> Vec<Result<StreamChunk, CompletionError>> {
        let provider = crate::OpenAiCompatibleProvider::new(crate::OpenAiCompatibleConfig {
            base_url,
            models: vec!["edge0_35b".to_owned()],
            ..Default::default()
        })
        .expect("provider");
        let stream = provider
            .invoke_model(
                "edge0_35b",
                infinity_provider_protocol::CompletionRequest {
                    chat_history: vec![infinity_provider_protocol::Message::user("hi")],
                    ..Default::default()
                },
            )
            .await
            .expect("stream");
        stream.collect().await
    }

    #[tokio::test]
    async fn streams_text_tool_calls_and_usage() {
        const BODY: &str = concat!(
            "data: {\"choices\":[{\"delta\":{\"role\":\"assistant\",\"reasoning_content\":\"thinking\"}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"content\":\"Hel\"}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"content\":\"lo\"}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_abc\",\"function\":{\"name\":\"read_file\",\"arguments\":\"{\\\"path\\\":\"}}]}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"\\\"main.rs\\\"}\"}}]}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n",
            "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":12,\"completion_tokens\":7,\"total_tokens\":19}}\n\n",
            "data: [DONE]\n\n",
        );
        let base_url = serve_sse(BODY).await;
        let chunks: Vec<StreamChunk> = collect(base_url)
            .await
            .into_iter()
            .map(|chunk| chunk.expect("chunk"))
            .collect();

        assert_eq!(
            chunks[0],
            StreamChunk::ReasoningDelta {
                id: None,
                text: "thinking".to_owned()
            }
        );
        assert_eq!(chunks[1], StreamChunk::Text("Hel".to_owned()));
        assert_eq!(chunks[2], StreamChunk::Text("lo".to_owned()));
        assert!(matches!(
            &chunks[3],
            StreamChunk::ToolCallDelta {
                content: ToolCallDeltaContent::Name(name),
                ..
            } if name == "read_file"
        ));
        let tool_call = chunks
            .iter()
            .find_map(|chunk| match chunk {
                StreamChunk::ToolCall(call) => Some(call),
                _ => None,
            })
            .expect("complete tool call");
        assert_eq!(tool_call.id, "call_abc");
        assert_eq!(tool_call.function.name, "read_file");
        assert_eq!(
            tool_call.function.arguments,
            serde_json::json!({"path": "main.rs"})
        );
        assert_eq!(
            chunks.last(),
            Some(&StreamChunk::Final(FinalResponse {
                usage: Some(Usage {
                    input_tokens: 12,
                    output_tokens: 7,
                    total_tokens: 19,
                    cached_input_tokens: 0,
                }),
            }))
        );
    }

    #[tokio::test]
    async fn ends_the_turn_without_done() {
        const BODY: &str = "data: {\"choices\":[{\"delta\":{\"content\":\"bye\"}}]}\n\n";
        let chunks: Vec<StreamChunk> = collect(serve_sse(BODY).await)
            .await
            .into_iter()
            .map(|chunk| chunk.expect("chunk"))
            .collect();
        assert_eq!(chunks[0], StreamChunk::Text("bye".to_owned()));
        assert!(matches!(
            chunks.last(),
            Some(StreamChunk::Final(FinalResponse { usage: None }))
        ));
    }

    #[tokio::test]
    async fn classifies_http_failures() {
        let busy = error_from_response(
            StatusCode::TOO_MANY_REQUESTS,
            r#"{"error":{"message":"one inference is running","code":"busy"}}"#,
        );
        assert_eq!(busy.class(), ErrorClass::Throttled);

        let overflow = error_from_response(
            StatusCode::BAD_REQUEST,
            r#"{"error":{"message":"input exceeds capacity: messages","code":"context_length_exceeded"}}"#,
        );
        assert_eq!(overflow.class(), ErrorClass::ContextOverflow);

        let too_large = error_from_response(
            StatusCode::PAYLOAD_TOO_LARGE,
            r#"{"error":{"message":"body too large","code":"request_too_large"}}"#,
        );
        assert_eq!(too_large.class(), ErrorClass::ContextOverflow);

        let shutdown = error_from_response(
            StatusCode::SERVICE_UNAVAILABLE,
            r#"{"error":{"message":"shutting down","code":"shutting_down"}}"#,
        );
        assert_eq!(shutdown.class(), ErrorClass::Transient);

        let bad_request = error_from_response(
            StatusCode::BAD_REQUEST,
            r#"{"error":{"message":"Unknown field: foo","code":"invalid_parameter"}}"#,
        );
        assert_eq!(bad_request.class(), ErrorClass::Fatal);

        let wrong_model = error_from_response(
            StatusCode::NOT_FOUND,
            r#"{"error":{"message":"wrong alias","code":"model_not_found"}}"#,
        );
        assert_eq!(wrong_model.class(), ErrorClass::Fatal);
    }

    #[tokio::test]
    async fn assembles_interleaved_parallel_tool_calls() {
        // Servers that honor `parallel_tool_calls` may interleave the
        // argument fragments of several calls, so a call is only complete at
        // `finish_reason`.
        const BODY: &str = concat!(
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_a\",\"function\":{\"name\":\"read_file\",\"arguments\":\"{\\\"path\\\":\"}}]}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":1,\"id\":\"call_b\",\"function\":{\"name\":\"list_dir\",\"arguments\":\"{\\\"dir\\\":\"}}]}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"\\\"a.rs\\\"}\"}}]}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":1,\"function\":{\"arguments\":\"\\\"src\\\"}\"}}]}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n",
            "data: [DONE]\n\n",
        );
        let chunks: Vec<StreamChunk> = collect(serve_sse(BODY).await)
            .await
            .into_iter()
            .map(|chunk| chunk.expect("chunk"))
            .collect();

        let calls: Vec<&ToolCall> = chunks
            .iter()
            .filter_map(|chunk| match chunk {
                StreamChunk::ToolCall(call) => Some(call),
                _ => None,
            })
            .collect();
        assert_eq!(calls.len(), 2, "both parallel calls complete: {chunks:?}");
        assert_eq!(calls[0].id, "call_a");
        assert_eq!(calls[0].function.name, "read_file");
        assert_eq!(
            calls[0].function.arguments,
            serde_json::json!({"path": "a.rs"})
        );
        assert_eq!(calls[1].id, "call_b");
        assert_eq!(calls[1].function.name, "list_dir");
        assert_eq!(
            calls[1].function.arguments,
            serde_json::json!({"dir": "src"})
        );
        assert!(matches!(chunks.last(), Some(StreamChunk::Final(_))));
    }

    #[tokio::test]
    async fn reports_a_server_error_inside_the_stream() {
        // `edge0-infer` can fail a request after streaming has begun; the
        // error arrives as an event, not as an HTTP status.
        const BODY: &str = concat!(
            "data: {\"choices\":[{\"delta\":{\"content\":\"partial\"}}]}\n\n",
            "data: {\"error\":{\"message\":\"one inference is running\",\"code\":\"busy\"}}\n\n",
        );
        let chunks = collect(serve_sse(BODY).await).await;

        assert_eq!(chunks.len(), 2, "no final chunk after a failed stream");
        assert_eq!(
            chunks[0].as_ref().expect("text"),
            &StreamChunk::Text("partial".to_owned())
        );
        let error = chunks[1].as_ref().expect_err("stream error");
        assert_eq!(error.class(), ErrorClass::Throttled);
        assert!(error.to_string().contains("one inference is running"));
    }

    #[test]
    fn invalid_tool_arguments_are_transient() {
        let mut calls = BTreeMap::new();
        calls.insert(
            0,
            PartialCall {
                id: "call_1".to_owned(),
                name: "read_file".to_owned(),
                arguments: "{\"path\": ".to_owned(),
                name_emitted: true,
            },
        );
        let chunks = finish_calls(&mut calls);
        assert_eq!(chunks.len(), 1);
        assert_eq!(
            chunks[0].as_ref().expect_err("error").class(),
            ErrorClass::Transient
        );
    }
}
