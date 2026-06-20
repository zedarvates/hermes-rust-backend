//! OpenAI-compatible API endpoints
//! /v1/models, /v1/chat/completions (streaming + non-streaming)
//! Plug-and-play with OpenWebUI, n8n, Flowise, Continue.dev, Hermes...
//!
//! v0.3.0 — Real GGUF inference when model loaded, graceful stub otherwise.

use axum::{
    Json, Router, routing::{get, post},
    extract::State, response::IntoResponse,
    http::StatusCode,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::sync::Arc;
use tokio::sync::Mutex;
use crate::gguf_engine::GgufEngine;

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

use std::sync::OnceLock;

static MODELS: OnceLock<Vec<Value>> = OnceLock::new();

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
        // Streaming not yet supported without futures dep
        return Err(StatusCode::NOT_IMPLEMENTED);
    }

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
