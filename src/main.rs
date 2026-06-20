use axum::{Router, Json, routing::get, response::IntoResponse, extract::State};
use serde_json::json;
use tower_http::cors::CorsLayer;
use tracing_subscriber;
use std::sync::Arc;
use tokio::sync::Mutex;

mod gguf_engine;
mod openai_api;
mod api_server;
mod auth;
mod hailo_engine;
mod multi_gpu;
mod onnx_engine;
mod mcp_bridge;

use gguf_engine::{GgufEngine, detect_gpu_backend};
use openai_api::AppState;

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();

    let (backend, gpu_count) = detect_gpu_backend();
    println!("🦀 Hermes Rust Backend v0.3.0");
    println!("   GPU Backend: {} ({} GPU(s))", backend, gpu_count);
    if backend == gguf_engine::GpuBackend::CUDA {
        println!("   Flash Attention: ✓ supported via LocalAI");
    }

    // Initialize GGUF engine as LocalAI proxy
    let mut engine = GgufEngine::new();
    gguf_engine::set_localai_url("http://192.168.1.47:8080");

    // Try to connect to LocalAI and list models
    match engine.refresh_models().await {
        Ok(()) => {
            let models = engine.list_models();
            println!("   ✅ Connected to LocalAI ({} models)", models.len());
            for m in models {
                println!("      - {}", m);
            }
        }
        Err(e) => {
            println!("   ⚠️ LocalAI unreachable: {}", e);
            println!("   💡 The backend will work as a proxy once LocalAI is available");
        }
    }

    // Set Hailo MCP URL
    hailo_engine::set_mcp_url("http://192.168.1.47:8767");

    let state = Arc::new(AppState {
        engine: Arc::new(Mutex::new(engine)),
    });

    let app = Router::new()
        .route("/health", get(health))
        .route("/engines", get(engines_info))
        .merge(openai_api::openai_routes())
        .merge(api_server::routes())
        .merge(mcp_bridge::mcp_routes())
        .layer(CorsLayer::permissive())
        .with_state(state);

    let addr = "0.0.0.0:8769";
    println!();
    println!("   🚀 Hermes Rust Backend ready");
    println!("   📍 http://{addr}");
    println!("   ├── /health");
    println!("   ├── /engines");
    println!("   ├── /v1/chat/completions (OpenAI-compatible)");
    println!("   ├── /v1/models");
    println!("   └── /mcp/tools (MCP protocol)");
    println!();
    println!("   👁️  Hailo-8:      http://192.168.1.47:8767");
    println!("   🧠 LocalAI:      http://192.168.1.47:8080");

    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

async fn health() -> impl IntoResponse {
    Json(json!({
        "service": "hermes-rust-backend",
        "version": "0.3.0",
        "status": "ok",
        "features": ["gguf-proxy", "openai_api", "hailo8", "mcp"],
        "localai": gguf_engine::detect_gpu_backend().0.to_string()
    }))
}

async fn engines_info(state: State<Arc<AppState>>) -> impl IntoResponse {
    let (backend, gpu_count) = detect_gpu_backend();
    let engine = state.engine.lock().await;
    Json(json!({
        "engines": {
            "gguf": {
                "available": engine.is_available(),
                "models": engine.list_models(),
                "backend": backend.to_string(),
                "gpu_count": gpu_count,
                "proxy": "LocalAI (192.168.1.47:8080)"
            },
            "hailo8": {
                "available": hailo_engine::is_available(),
                "device": "/dev/hailo0",
                "mcp_bridge": "http://192.168.1.47:8767"
            },
            "cuda": {
                "available": multi_gpu::is_available(),
                "device_count": multi_gpu::device_count()
            },
            "onnx": {
                "available": onnx_engine::is_available()
            }
        }
    }))
}
