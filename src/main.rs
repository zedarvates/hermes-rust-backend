use axum::{Router, Json, routing::get, response::IntoResponse, extract::State};
use serde_json::json;
use tower_http::cors::{CorsLayer, Any};
use tracing_subscriber;
use std::sync::Arc;
use tokio::sync::Mutex;

mod gguf_engine;
mod openai_api;
mod api_server;
mod auth;

use gguf_engine::{GgufEngine, detect_gpu_backend};
use openai_api::AppState;

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();

    let (backend, gpu_count) = detect_gpu_backend();
    println!("🦀 Hermes Rust Backend v0.2.0");
    println!("   GPU Backend: {} ({} GPU(s))", backend, gpu_count);
    if backend == gguf_engine::GpuBackend::CUDA {
        println!("   Flash Attention: enabled");
    }

    let engine = GgufEngine::new();
    let state = Arc::new(AppState {
        engine: Arc::new(Mutex::new(engine)),
    });

    let app = Router::new()
        .route("/health", get(health))
        .route("/engines", get(engines_info))
        .merge(openai_api::openai_routes())
        .merge(api_server::api_routes())
        .layer(CorsLayer::permissive())
        .with_state(state);

    let addr = "0.0.0.0:8769";
    println!("   Listening on {addr}");
    println!("   API: http://{addr}/v1/chat/completions");
    println!("   Health: http://{addr}/health");
    println!("   Load a .gguf model to enable inference");

    axum::serve(
        tokio::net::TcpListener::bind(addr).await.unwrap(),
        app,
    ).await.unwrap();
}

async fn health() -> impl IntoResponse {
    Json(json!({
        "service": "hermes-rust-backend",
        "version": "0.2.0",
        "status": "ok",
        "features": ["gguf", "openai_api", "hailo8", "flash_attention"]
    }))
}

async fn engines_info(state: State<Arc<AppState>>) -> impl IntoResponse {
    let (backend, gpu_count) = detect_gpu_backend();
    let engine = state.engine.lock().await;
    Json(json!({
        "engines": {
            "gguf": {
                "available": engine.is_available(),
                "backend": backend.to_string(),
                "gpu_count": gpu_count,
                "flash_attention": backend == gguf_engine::GpuBackend::CUDA
            },
            "hailo8": {
                "available": std::path::Path::new("/dev/hailo0").exists()
            }
        }
    }))
}
