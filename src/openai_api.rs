//! OpenAI-compatible API endpoints
//! /v1/models, /v1/chat/completions
//! Plug-and-play with OpenWebUI, n8n, Flowise, Continue.dev, Hermes...

use axum::{Json, Router, routing::{get, post}, extract::State};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::Mutex;
use crate::gguf_engine::{GgufEngine, InferenceResult};

/// Shared application state
pub struct AppState {
    pub engine: Arc<Mutex<GgufEngine>>,
}

#[derive(Debug, Serialize)]
pub struct ModelInfo {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub owned_by: String,
}

#[derive(Debug, Serialize)]
pub struct ModelsResponse {
    pub object: String,
    pub data: Vec<ModelInfo>,
}

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
pub struct ChatCompletionChoice {
    pub index: u32,
    pub message: ChatMessage,
    pub finish_reason: String,
}

#[derive(Debug, Serialize)]
pub struct UsageInfo {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

#[derive(Debug, Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatCompletionChoice>,
    pub usage: UsageInfo,
}

pub fn openai_routes() -> Router<Arc<AppState>> {
    Router::new()
        .route("/v1/models", get(list_models))
        .route("/v1/chat/completions", post(chat_completions))
}

async fn list_models() -> Json<ModelsResponse> {
    let models = vec![
        ModelInfo { id: "gguf".into(), object: "model".into(), created: 1714867200, owned_by: "hermes".into() },
        ModelInfo { id: "hailo8-vision".into(), object: "model".into(), created: 1714867200, owned_by: "hermes".into() },
    ];
    Json(ModelsResponse { object: "list".into(), data: models })
}

async fn chat_completions(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ChatCompletionRequest>,
) -> Json<ChatCompletionResponse> {
    // Build prompt from messages
    let prompt = req.messages.iter()
        .map(|m| format!("[{}] {}", m.role, m.content))
        .collect::<Vec<_>>()
        .join("\n");

    let engine = state.engine.lock().await;

    let (content, usage) = if engine.is_available() {
        // Real GGUF inference
        match engine.infer(&prompt, req.max_tokens).await {
            Ok(result) => {
                let tokens = result.tokens_generated;
                (result.text, UsageInfo {
                    prompt_tokens: 0, // tokenizer would give exact count
                    completion_tokens: tokens,
                    total_tokens: tokens,
                })
            }
            Err(e) => {
                (format!("[Inference error: {}]", e), UsageInfo {
                    prompt_tokens: 0, completion_tokens: 0, total_tokens: 0,
                })
            }
        }
    } else {
        // No model loaded — friendly message
        (format!(
            "🦀 Hermes Rust Backend v0.2.0\nNo GGUF model loaded. Drop a .gguf file in models/ and call load_model().\nYour message: {}",
            req.messages.last().map(|m| m.content.as_str()).unwrap_or("")
        ), UsageInfo {
            prompt_tokens: 0, completion_tokens: 0, total_tokens: 0,
        })
    };

    drop(engine);

    Json(ChatCompletionResponse {
        id: format!("chatcmpl-{}", std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs()),
        object: "chat.completion".into(),
        created: 1714867200,
        model: req.model,
        choices: vec![ChatCompletionChoice {
            index: 0,
            message: ChatMessage {
                role: "assistant".into(),
                content,
            },
            finish_reason: "stop".into(),
        }],
        usage,
    })
}
