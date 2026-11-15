//! Conversion from Infinity's provider protocol to OpenAI Chat Completions
//! request bodies.

use infinity_provider_protocol as proto;
use serde_json::{Map, Value, json};

/// Build the JSON body for `POST /chat/completions`.
///
/// The request always streams and asks for a final usage chunk. Tool
/// definitions are sent in the OpenAI function format; local servers such as
/// Edge0's translate them into the model's chat template. `parallel_tool_calls`
/// is always sent when tools are present: Infinity executes only the first tool
/// call of a turn, so asking for a single call keeps the history consistent.
pub fn build_chat_request(
    model_id: &str,
    request: &proto::CompletionRequest,
    parallel_tool_calls: bool,
) -> Result<Value, proto::CompletionError> {
    let messages = messages_to_openai(&request.preamble, &request.chat_history)?;

    let mut body = Map::new();
    body.insert("model".to_owned(), json!(model_id));
    body.insert("messages".to_owned(), Value::Array(messages));
    body.insert("stream".to_owned(), json!(true));
    body.insert(
        "stream_options".to_owned(),
        json!({ "include_usage": true }),
    );

    if !request.tools.is_empty() {
        body.insert(
            "tools".to_owned(),
            Value::Array(
                request
                    .tools
                    .iter()
                    .map(|tool| {
                        json!({
                            "type": "function",
                            "function": {
                                "name": tool.name,
                                "description": tool.description,
                                "parameters": tool.parameters,
                            }
                        })
                    })
                    .collect(),
            ),
        );
        body.insert("tool_choice".to_owned(), json!("auto"));
        body.insert("parallel_tool_calls".to_owned(), json!(parallel_tool_calls));
    }

    if let Some(max_tokens) = request.max_tokens.filter(|n| *n > 0) {
        body.insert("max_tokens".to_owned(), json!(max_tokens));
    }

    if let Some(additional) = &request.additional_params {
        let additional = additional.as_object().ok_or_else(|| {
            proto::CompletionError::RequestError("additional_params must be a JSON object".into())
        })?;
        for (key, value) in additional {
            // These are owned by the provider: overriding them would break the
            // streaming contract the runtime relies on.
            if matches!(
                key.as_str(),
                "model" | "messages" | "stream" | "stream_options"
            ) {
                continue;
            }
            body.insert(key.clone(), value.clone());
        }
    }

    Ok(Value::Object(body))
}

/// Convert the system prompt and conversation into Chat Completions messages.
///
/// Tool results become `role: "tool"` messages (one per result, before any
/// text of the same user turn), assistant tool calls become `tool_calls`, and
/// assistant reasoning is carried back as `reasoning_content`, the assistant
/// history extension the local servers understand. Images are not supported by
/// this provider (`supports_image_input` is false, so the runtime replaces
/// them with a placeholder before they reach us).
pub fn messages_to_openai(
    preamble: &Option<String>,
    chat_history: &[proto::Message],
) -> Result<Vec<Value>, proto::CompletionError> {
    if chat_history.is_empty() {
        return Err(proto::CompletionError::RequestError(
            "completion request has an empty chat history".into(),
        ));
    }

    let mut messages = Vec::new();
    if let Some(preamble) = preamble.as_deref()
        && !preamble.trim().is_empty()
    {
        messages.push(json!({ "role": "system", "content": preamble }));
    }

    for message in chat_history {
        match message {
            proto::Message::User { content } => {
                for block in content {
                    if let proto::UserContent::ToolResult(result) = block {
                        messages.push(json!({
                            "role": "tool",
                            "tool_call_id": tool_call_id(result)?,
                            "content": tool_result_text(result),
                        }));
                    }
                }
                let text = user_text(content);
                if !text.is_empty() {
                    messages.push(json!({ "role": "user", "content": text }));
                }
            }
            proto::Message::Assistant { content } => {
                let text = assistant_text(content);
                let reasoning = assistant_reasoning(content);
                let tool_calls = assistant_tool_calls(content)?;
                if text.is_empty() && tool_calls.is_empty() {
                    // Nothing the model can act on (for example a bare
                    // reasoning block); dropping it keeps the transcript
                    // valid for templates that reject empty assistant turns.
                    continue;
                }
                let mut message = Map::new();
                message.insert("role".to_owned(), json!("assistant"));
                message.insert(
                    "content".to_owned(),
                    if text.is_empty() {
                        Value::Null
                    } else {
                        json!(text)
                    },
                );
                if !tool_calls.is_empty() {
                    message.insert("tool_calls".to_owned(), Value::Array(tool_calls));
                }
                if let Some(reasoning) = reasoning {
                    message.insert("reasoning_content".to_owned(), json!(reasoning));
                }
                messages.push(Value::Object(message));
            }
        }
    }

    Ok(messages)
}

fn user_text(content: &[proto::UserContent]) -> String {
    content
        .iter()
        .filter_map(|block| match block {
            proto::UserContent::Text(text) => Some(text.text.as_str()),
            proto::UserContent::ToolResult(_) | proto::UserContent::Image(_) => None,
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn assistant_text(content: &[proto::AssistantContent]) -> String {
    content
        .iter()
        .filter_map(|block| match block {
            proto::AssistantContent::Text(text) => Some(text.text.as_str()),
            proto::AssistantContent::ToolCall(_) | proto::AssistantContent::Reasoning(_) => None,
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn assistant_reasoning(content: &[proto::AssistantContent]) -> Option<String> {
    let text = content
        .iter()
        .filter_map(|block| match block {
            proto::AssistantContent::Reasoning(reasoning) => Some(reasoning.display_text()),
            proto::AssistantContent::Text(_) | proto::AssistantContent::ToolCall(_) => None,
        })
        .filter(|text| !text.is_empty())
        .collect::<Vec<_>>()
        .join("\n");
    (!text.is_empty()).then_some(text)
}

fn assistant_tool_calls(
    content: &[proto::AssistantContent],
) -> Result<Vec<Value>, proto::CompletionError> {
    content
        .iter()
        .filter_map(|block| match block {
            proto::AssistantContent::ToolCall(call) => Some(call),
            proto::AssistantContent::Text(_) | proto::AssistantContent::Reasoning(_) => None,
        })
        .map(|call| {
            Ok(json!({
                "id": call_id(&call.id, call.call_id.as_deref())
                    .ok_or_else(|| proto::CompletionError::RequestError(
                        "assistant tool call has neither an id nor a call_id".into(),
                    ))?,
                "type": "function",
                "function": {
                    "name": call.function.name,
                    "arguments": serde_json::to_string(&call.function.arguments)?,
                }
            }))
        })
        .collect()
}

/// OpenAI matches results to calls by id. Infinity's history carries the
/// provider call id in `id` and, for providers that use a secondary id, in
/// `call_id`; fall back to whichever one is present.
fn tool_call_id(result: &proto::ToolResult) -> Result<String, proto::CompletionError> {
    call_id(&result.id, result.call_id.as_deref()).ok_or_else(|| {
        proto::CompletionError::RequestError(
            "tool result has neither an id nor a call_id to answer".into(),
        )
    })
}

fn call_id(id: &str, call_id: Option<&str>) -> Option<String> {
    if !id.is_empty() {
        Some(id.to_owned())
    } else {
        call_id.filter(|id| !id.is_empty()).map(str::to_owned)
    }
}

fn tool_result_text(result: &proto::ToolResult) -> String {
    let text = result
        .content
        .iter()
        .filter_map(|block| match block {
            proto::ToolResultContent::Text(text) => Some(text.text.as_str()),
            proto::ToolResultContent::Image(_) => None,
        })
        .collect::<Vec<_>>()
        .join("\n");
    if text.is_empty() {
        "(no output)".to_owned()
    } else {
        text
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proto::{
        AssistantContent, Message, Reasoning, ToolCall, ToolDefinition, ToolResult,
        ToolResultContent, UserContent,
    };

    fn request(chat_history: Vec<Message>) -> proto::CompletionRequest {
        proto::CompletionRequest {
            preamble: Some("be brief".to_owned()),
            chat_history,
            ..Default::default()
        }
    }

    #[test]
    fn maps_system_user_and_tools() {
        let body = build_chat_request(
            "edge0_35b",
            &proto::CompletionRequest {
                tools: vec![ToolDefinition {
                    name: "read_file".to_owned(),
                    description: "Read a file".to_owned(),
                    parameters: json!({"type": "object", "properties": {"path": {"type": "string"}}}),
                }],
                max_tokens: Some(256),
                ..request(vec![Message::user("hello")])
            },
            false,
        )
        .expect("build body");

        assert_eq!(body["model"], "edge0_35b");
        assert_eq!(body["stream"], true);
        assert_eq!(body["max_tokens"], 256);
        assert_eq!(body["parallel_tool_calls"], false);
        assert_eq!(body["tool_choice"], "auto");
        assert_eq!(body["tools"][0]["type"], "function");
        assert_eq!(body["tools"][0]["function"]["name"], "read_file");
        assert_eq!(body["messages"][0]["role"], "system");
        assert_eq!(body["messages"][0]["content"], "be brief");
        assert_eq!(body["messages"][1]["role"], "user");
        assert_eq!(body["messages"][1]["content"], "hello");
        assert!(
            body.get("tool_choice").is_some() && body.get("max_completion_tokens").is_none(),
            "only max_tokens is sent; OpenAI-compatible servers reject both"
        );
    }

    #[test]
    fn omits_tools_when_none_are_offered() {
        let body = build_chat_request("m", &request(vec![Message::user("hi")]), false)
            .expect("build body");
        assert!(body.get("tools").is_none());
        assert!(body.get("tool_choice").is_none());
        assert!(body.get("max_tokens").is_none());
    }

    #[test]
    fn maps_tool_calls_and_results() {
        let history = vec![
            Message::user("read main.rs"),
            Message::Assistant {
                content: vec![
                    AssistantContent::Reasoning(Reasoning::new_with_signature(
                        "need the file",
                        None,
                    )),
                    AssistantContent::ToolCall(ToolCall::new(
                        "call-1",
                        "read_file",
                        json!({"path": "main.rs"}),
                    )),
                ],
            },
            Message::User {
                content: vec![UserContent::ToolResult(ToolResult {
                    id: "call-1".to_owned(),
                    call_id: None,
                    content: vec![ToolResultContent::text("fn main() {}")],
                })],
            },
        ];
        let messages = messages_to_openai(&None, &history).expect("convert");

        assert_eq!(messages[1]["role"], "assistant");
        assert_eq!(messages[1]["content"], Value::Null);
        assert_eq!(messages[1]["reasoning_content"], "need the file");
        assert_eq!(messages[1]["tool_calls"][0]["id"], "call-1");
        assert_eq!(
            messages[1]["tool_calls"][0]["function"]["arguments"],
            "{\"path\":\"main.rs\"}"
        );
        assert_eq!(messages[2]["role"], "tool");
        assert_eq!(messages[2]["tool_call_id"], "call-1");
        assert_eq!(messages[2]["content"], "fn main() {}");
    }

    #[test]
    fn tool_results_precede_user_text_in_one_turn() {
        let history = vec![Message::User {
            content: vec![
                UserContent::text("here is the result"),
                UserContent::ToolResult(ToolResult {
                    id: "call-2".to_owned(),
                    call_id: None,
                    content: vec![ToolResultContent::text("ok")],
                }),
            ],
        }];
        let messages = messages_to_openai(&None, &history).expect("convert");
        assert_eq!(messages[0]["role"], "tool");
        assert_eq!(messages[1]["role"], "user");
        assert_eq!(messages[1]["content"], "here is the result");
    }

    #[test]
    fn empty_history_is_rejected() {
        let error = messages_to_openai(&None, &[]).expect_err("should fail");
        assert!(error.to_string().contains("empty chat history"));
    }

    #[test]
    fn additional_params_are_merged_but_cannot_override_the_stream() {
        let body = build_chat_request(
            "m",
            &proto::CompletionRequest {
                additional_params: Some(json!({
                    "temperature": 0.2,
                    "model": "other",
                    "stream": false,
                })),
                ..request(vec![Message::user("hi")])
            },
            false,
        )
        .expect("build body");
        assert_eq!(body["temperature"], 0.2);
        assert_eq!(body["model"], "m");
        assert_eq!(body["stream"], true);
    }
}
