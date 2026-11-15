//! End-to-end test of the provider binary: the daemon-to-provider contract
//! (socket path on the first stdout line) plus a streamed completion against a
//! stub OpenAI-compatible server.

use std::process::Stdio;

use axum::Router;
use axum::response::IntoResponse;
use axum::routing::post;
use futures_util::StreamExt;
use infinity_provider_protocol::remote::RemoteModelProvider;
use infinity_provider_protocol::{CompletionRequest, Message, ModelProvider as _, StreamChunk};
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::net::TcpListener;
use tokio::process::Command;

const SSE_BODY: &str = concat!(
    "data: {\"choices\":[{\"delta\":{\"content\":\"hi from edge0\"}}]}\n\n",
    "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
    "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":4,\"completion_tokens\":3,\"total_tokens\":7}}\n\n",
    "data: [DONE]\n\n",
);

async fn stub_server() -> String {
    async fn completions() -> impl IntoResponse {
        (
            [
                ("content-type", "text/event-stream"),
                ("cache-control", "no-cache"),
            ],
            SSE_BODY,
        )
    }

    let app = Router::new().route("/v1/chat/completions", post(completions));
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind stub");
    let address = listener.local_addr().expect("stub address");
    tokio::spawn(async move {
        axum::serve(listener, app).await.expect("serve stub");
    });
    format!("http://{address}/v1")
}

#[tokio::test]
async fn serves_a_completion_over_the_provider_socket() {
    let base_url = stub_server().await;

    let mut child = Command::new(env!("CARGO_BIN_EXE_infinity-provider-openai"))
        .args([
            "--base-url",
            &base_url,
            "--model",
            "edge0_35b",
            "--context-window",
            "16384",
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .kill_on_drop(true)
        .spawn()
        .expect("spawn provider");

    let stdout = child.stdout.take().expect("stdout");
    let mut line = String::new();
    BufReader::new(stdout)
        .read_line(&mut line)
        .await
        .expect("read socket path");
    let socket_path = line.trim();
    assert!(!socket_path.is_empty(), "provider printed no socket path");

    let provider = RemoteModelProvider::new(socket_path);
    let models = provider.list_models().await.expect("list models");
    assert_eq!(models.len(), 1);
    assert_eq!(models[0].model_id, "edge0_35b");
    assert_eq!(models[0].context_window, 16384);

    let stream = provider
        .invoke_model(
            "edge0_35b",
            CompletionRequest {
                chat_history: vec![Message::user("hello")],
                ..Default::default()
            },
        )
        .await
        .expect("invoke model");
    let chunks: Vec<StreamChunk> = stream
        .map(|chunk| chunk.expect("chunk"))
        .collect::<Vec<_>>()
        .await;

    assert!(chunks.contains(&StreamChunk::Text("hi from edge0".to_owned())));
    assert!(matches!(chunks.last(), Some(StreamChunk::Final(_))));

    child.kill().await.expect("stop provider");
}
