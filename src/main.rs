use axum::{Router, Json, routing::get, response::IntoResponse, extract::State};
use serde_json::json;
use tower_http::cors::CorsLayer;
use tracing_subscriber;
use std::sync::Arc;
use tokio::sync::Mutex;
use std::path::Path;

mod gguf_engine;
mod openai_api;
mod api_server;
mod auth;
mod hailo_engine;
mod multi_gpu;
mod onnx_engine;
mod mcp_bridge;

use gguf_engine::{GgufEngine, detect_gpu_backend, GgufModelConfig};
use openai_api::AppState;

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();

    let (backend, gpu_count) = detect_gpu_backend();
    println!("🦀 Hermes Rust Backend v0.3.0");
    println!("   GPU Backend: {} ({} GPU(s))", backend, gpu_count);
    if backend == gguf_engine::GpuBackend::CUDA {
        println!("   Flash Attention: ✓ enabled");
    }

    // Auto-discover GGUF models
    let mut engine = GgufEngine::new();
    let models_dir = Path::new("models");
    if models_dir.exists() {
        let mut loaded = false;
        if let Ok(entries) = std::fs::read_dir(models_dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().map(|e| e == "gguf").unwrap_or(false) {
                    let model_name = path.file_stem()
                        .and_then(|s| s.to_str())
                        .unwrap_or("unknown")
                        .to_string();
                    println!("   📦 Found GGUF model: {} ({})", model_name, path.display());

                    let config = GgufModelConfig {
                        model_path: path,
                        n_gpu_layers: 999,
                        main_gpu: 0,
                        flash_attention: backend == gguf_engine::GpuBackend::CUDA,
                        n_ctx: 8192,
                        n_batch: 512,
                        temperature: 0.7,
                        max_tokens: 4096,
                    };

                    match engine.load_model(config) {
                        Ok(_) => {
                            println!("   ✅ Model loaded: {}", model_name);
                            loaded = true;
                            break; // Load first model only
                        }
                        Err(e) => {
                            println!("   ⚠️ Failed to load {}: {}", model_name, e);
                        }
                    }
                }
            }
        }
        if !loaded {
            println!("   ⚠️ No GGUF models could be loaded from models/");
        }
    } else {
        println!("   📁 No models/ directory found. Create one with .gguf files.");
        println!("   💡 Use: mkdir models && cp your-model.gguf models/");
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
        .merge(api_server::api_routes())
        .merge(mcp_bridge::mcp_routes())
        .layer(CorsLayer::permissive())
        .with_state(state);

    let addr = "0.0.0.0:8769";
    println!("   Listening on {addr}");
    println!("   OpenAI API: http://{addr}/v1/chat/completions");
    println!("   Streaming:  http://{addr}/v1/chat/completions?stream=true");
    println!("   MCP:        http://{addr}/mcp/tools");
    println!("   Health:     http://{addr}/health");
    println!("   Hailo-8:    {} (bridge: 192.168.1.47:8767)", 
        if hailo_engine::is_available() { "✓ available" } else { "not detected" });

    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

async fn health() -> impl IntoResponse {
    Json(json!({
        "service": "hermes-rust-backend",
        "version": "0.3.0",
        "status": "ok",
        "features": ["gguf", "openai_api", "hailo8", "flash_attention", "mcp", "streaming"]
    }))
}

async fn engines_info(state: State<Arc<AppState>>) -> impl IntoResponse {
    let (backend, gpu_count) = detect_gpu_backend();
    let engine = state.engine.lock().await;
    Json(json!({
        "engines": {
            "gguf": {
                "available": engine.is_available(),
                "model_loaded": engine.model_loaded,
                "backend": backend.to_string(),
                "gpu_count": gpu_count,
                "flash_attention": backend == gguf_engine::GpuBackend::CUDA
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
