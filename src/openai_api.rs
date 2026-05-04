//! OpenAI-compatible API endpoints
//! /v1/models, /v1/chat/completions
//! Plug-and-play with OpenWebUI, n8n, Flowise, Continue.dev, Hermes...

use axum::{Json, Router, routing::{get, post}, extract::State};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::Mutex;
use crate::gguf_engine::GgufEngine;

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

pub fn openai_routes() -> Router {
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
    Json(req): Json<ChatCompletionRequest>,
) -> Json<ChatCompletionResponse> {
    let user_msg = req.messages.last()
        .map(|m| m.content.clone())
        .unwrap_or_default();
    Json(ChatCompletionResponse {
        id: "chatcmpl-001".into(),
        object: "chat.completion".into(),
        created: 1714867200,
        model: req.model,
        choices: vec![ChatCompletionChoice {
            index: 0,
            message: ChatMessage {
                role: "assistant".into(),
                content: format!("[Hermes Rust Backend] GGUF engine ready. Load a .gguf model to start. Your message: {}", user_msg),
            },
            finish_reason: "stop".into(),
        }],
        usage: UsageInfo { prompt_tokens: 10, completion_tokens: 20, total_tokens: 30 },
    })
}
