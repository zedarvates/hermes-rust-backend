//! OpenAI-compatible API endpoints
//! /v1/models, /v1/chat/completions (streaming + non-streaming)
//! Plug-and-play with OpenWebUI, n8n, Flowise, Continue.dev, Hermes...
//!
//! v0.3.0 — Real GGUF inference when model loaded, graceful stub otherwise.

use axum::{
    Json, Router, routing::{get, post},
    extract::State, response::{IntoResponse, Sse},
    http::StatusCode,
};
use futures::stream::{self, Stream};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::sync::Arc;
use std::convert::Infallible;
use tokio::sync::Mutex;
use crate::gguf_engine::{GgufEngine, InferenceResult};

/// Shared application state
pub struct AppState {
    pub engine: Arc<Mutex<GgufEngine>>,
}

// ── Request / Response types ─────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: u32,
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    #[serde(default)]
    pub stream: bool,
}

fn default_max_tokens() -> u32 { 2048 }
fn default_temperature() -> f32 { 0.7 }

#[derive(Debug, Serialize)]
pub struct UsageInfo {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

// ── Model registry ──────────────────────────────────────────

static MODELS: once_cell::sync::Lazy<Vec<Value>> = once_cell::sync::Lazy::new(|| {
    vec![
        json!({"id": "Qwen3.6-27B-IQ4_XS", "object": "model", "owned_by": "hermes-rust-backend"}),
        json!({"id": "Qwen3.6-27B-UD-Q8_K_XL", "object": "model", "owned_by": "hermes-rust-backend"}),
        json!({"id": "gguf", "object": "model", "owned_by": "hermes-rust-backend"}),
        json!({"id": "hailo8-vision", "object": "model", "owned_by": "hermes-rust-backend"}),
    ]
});

// ── Routes ──────────────────────────────────────────────────

pub fn openai_routes() -> Router<Arc<AppState>> {
    Router::new()
        .route("/v1/models", get(list_models))
        .route("/v1/chat/completions", post(chat_completions))
}

// ── Handlers ─────────────────────────────────────────────────

async fn list_models() -> impl IntoResponse {
    #[derive(Serialize)]
    struct ModelsResponse {
        object: String,
        data: Vec<Value>,
    }
    Json(ModelsResponse {
        object: "list".into(),
        data: MODELS.clone(),
    })
}

async fn chat_completions(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ChatCompletionRequest>,
) -> Result<impl IntoResponse, StatusCode> {
    // Build prompt from messages (simple chat format)
    let prompt = req.messages.iter()
        .map(|m| format!("<|{}|>\n{}", m.role, m.content))
        .collect::<Vec<_>>()
        .join("\n");

    let engine = state.engine.lock().await;

    if req.stream {
        // Streaming mode (SSE)
        if !engine.is_available() {
            drop(engine);
            return Ok(Sse::new(stream::once(async {
                Ok::<_, Infallible>(axum::response::sse::Event::default()
                    .data(json!({"error": "No GGUF model loaded"}).to_string()))
            })).into_response());
        }

        let prompt_clone = prompt.clone();
        let max_tokens = req.max_tokens;

        let stream = stream::unfold(
            (state.clone(), prompt_clone, max_tokens, 0u32),
            move |(st, p, max, pos)| async move {
                if pos >= max {
                    // Send [DONE] signal
                    return Some((Ok::<_, Infallible>(
                        axum::response::sse::Event::default().data("[DONE]")
                    ), (st, p, max, pos)));
                }

                let eng = st.engine.lock().await;
                match eng.infer(&p, 1).await {
                    Ok(result) => {
                        let chunk = json!({
                            "choices": [{
                                "delta": {"content": result.text},
                                "index": 0
                            }]
                        });
                        drop(eng);
                        Some((Ok::<_, Infallible>(
                            axum::response::sse::Event::default()
                                .data(chunk.to_string())
                        ), (st, p, max, pos + 1)))
                    }
                    Err(e) => {
                        drop(eng);
                        let err_chunk = json!({"error": e.to_string()});
                        Some((Ok::<_, Infallible>(
                            axum::response::sse::Event::default()
                                .data(err_chunk.to_string())
                        ), (st, p, max, max))) // stop
                    }
                }
            }
        );

        Ok(Sse::new(stream).into_response())
    } else {
        // Non-streaming mode
        if engine.is_available() {
            match engine.infer(&prompt, req.max_tokens).await {
                Ok(result) => {
                    let tokens = result.tokens_generated;
                    drop(engine);
                    Ok(Json(json!({
                        "id": format!("chatcmpl-{}", chrono::Utc::now().timestamp()),
                        "object": "chat.completion",
                        "created": chrono::Utc::now().timestamp(),
                        "model": req.model,
                        "choices": [{
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": result.text
                            },
                            "finish_reason": "stop"
                        }],
                        "usage": {
                            "prompt_tokens": 0,
                            "completion_tokens": tokens,
                            "total_tokens": tokens
                        }
                    })).into_response())
                }
                Err(e) => {
                    drop(engine);
                    Err(StatusCode::INTERNAL_SERVER_ERROR)
                }
            }
        } else {
            // No model loaded — friendly stub with model info
            let last_msg = req.messages.last()
                .map(|m| m.content.as_str())
                .unwrap_or("");
            drop(engine);
            Ok(Json(json!({
                "id": format!("chatcmpl-{}", chrono::Utc::now().timestamp()),
                "object": "chat.completion",
                "created": chrono::Utc::now().timestamp(),
                "model": req.model,
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": format!(
                            "Hermes Rust Backend v0.3.0\n\
                             GGUF Engine ready. 2 model(s) available. \
                             Your message: \"{}\". Full inference coming soon.",
                            &last_msg[..last_msg.len().min(50)]
                        )
                    },
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            })).into_response())
        }
    }
}
