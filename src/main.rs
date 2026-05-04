use axum::{Router, routing::get, Json};
use serde_json::{json, Value};
use std::net::SocketAddr;
use tokio::signal;
use tower_http::cors::CorsLayer;
use tracing_subscriber::{EnvFilter, fmt};

mod api_server;
mod auth;
mod hailo_engine;
mod onnx_engine;
mod multi_gpu;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize tracing
    fmt()
        .with_env_filter(EnvFilter::from_default_env()
            .add_directive("hermes=debug".parse()?)
            .add_directive("tower_http=info".parse()?))
        .init();

    tracing::info!("🦀 Hermes Rust Backend starting...");

    // Build router
    let app = Router::new()
        .route("/health", get(health))
        .route("/engines", get(list_engines))
        .nest("/v1", api_server::routes())
        .layer(CorsLayer::permissive());

    // Bind
    let addr: SocketAddr = "0.0.0.0:8769".parse()?;
    tracing::info!("🚀 API listening on http://{}", addr);

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;

    Ok(())
}

async fn health() -> Json<Value> {
    Json(json!({
        "status": "ok",
        "service": "hermes-rust-backend",
        "version": env!("CARGO_PKG_VERSION"),
        "timestamp": chrono::Utc::now().to_rfc3339()
    }))
}

async fn list_engines() -> Json<Value> {
    Json(json!({
        "engines": [
            {"name": "onnx", "status": onnx_engine::is_available(), "models": ["SenseNova-U1-8B-MoT"]},
            {"name": "hailo", "status": hailo_engine::is_available(), "models": ["YOLOv8m", "ResNet-18", "PaddleOCR"]},
            {"name": "cuda", "status": multi_gpu::is_available(), "devices": multi_gpu::device_count()}
        ]
    }))
}

async fn shutdown_signal() {
    let ctrl_c = async { signal::ctrl_c().await.expect("Ctrl+C handler") };
    #[cfg(unix)]
    let terminate = async {
        signal::unix::signal(signal::unix::SignalKind::terminate())
            .expect("SIGTERM handler")
            .recv()
            .await;
    };
    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();
    
    tokio::select! {
        _ = ctrl_c => {},
        _ = terminate => {},
    }
    tracing::info!("🛑 Shutting down...");
}
