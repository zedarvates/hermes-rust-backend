use axum::{Router, Json, routing::get, response::IntoResponse};
use serde_json::json;
use tower_http::cors::{CorsLayer, Any};
use tracing_subscriber;

mod gguf_engine;
mod openai_api;
mod api_server;
mod auth;

use gguf_engine::{GgufEngine, detect_gpu_backend};

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();

    let (backend, gpu_count) = detect_gpu_backend();
    println!("🦀 Hermes Rust Backend v0.2.0");
    println!("   GPU Backend: {} ({} GPU(s))", backend, gpu_count);
    if backend == gguf_engine::GpuBackend::CUDA {
        println!("   Flash Attention: enabled");
    }

    let gguf = GgufEngine::new();
    let state = axum::extract::State(std::sync::Arc::new(std::sync::Mutex::new(gguf)));

    let app = Router::new()
        .route("/health", get(health))
        .route("/engines", get(engines_info))
        .merge(openai_api::openai_routes())
        .merge(api_server::api_routes())
        .layer(CorsLayer::permissive());

    let addr = "0.0.0.0:8769";
    println!("   Listening on {addr}");
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

async fn engines_info() -> impl IntoResponse {
    let (backend, gpu_count) = detect_gpu_backend();
    Json(json!({
        "engines": {
            "gguf": {
                "available": true,
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
