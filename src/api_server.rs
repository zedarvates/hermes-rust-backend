use axum::{Router, routing::{get, post}, Json, extract::State, http::StatusCode};
use serde_json::{json, Value};
use std::sync::Arc;

pub struct AppState {
    pub start_time: chrono::DateTime<chrono::Utc>,
}

pub fn routes() -> Router {
    let state = Arc::new(AppState {
        start_time: chrono::Utc::now(),
    });
    
    Router::new()
        .route("/status", get(status))
        .route("/infer", post(infer))
        .route("/ocr", post(ocr))
        .route("/detect", post(detect))
        .with_state(state)
}

async fn status(State(state): State<Arc<AppState>>) -> Json<Value> {
    let uptime = chrono::Utc::now() - state.start_time;
    Json(json!({
        "uptime_seconds": uptime.num_seconds(),
        "engines": {
            "onnx": super::onnx_engine::is_available(),
            "hailo": super::hailo_engine::is_available(),
            "cuda": super::multi_gpu::is_available(),
        },
        "gpu_count": super::multi_gpu::device_count()
    }))
}

async fn infer(State(_state): State<Arc<AppState>>, Json(payload): Json<Value>) -> Result<Json<Value>, StatusCode> {
    let model = payload.get("model").and_then(|v| v.as_str()).unwrap_or("sensenova-u1");
    let prompt = payload.get("prompt").and_then(|v| v.as_str()).unwrap_or("");
    
    tracing::info!("Inference request: model={}, prompt_len={}", model, prompt.len());
    
    Ok(Json(json!({
        "status": "ok",
        "model": model,
        "tokens": 0,
        "output": format!("[Placeholder] Model {} would process: {}", model, &prompt[..prompt.len().min(50)]),
        "engine": "rust-backend"
    })))
}

async fn ocr(Json(_payload): Json<Value>) -> Result<Json<Value>, StatusCode> {
    Ok(Json(json!({
        "status": "ok",
        "text": "[OCR placeholder - Hailo-8 PaddleOCR]",
        "engine": "hailo-8"
    })))
}

async fn detect(Json(_payload): Json<Value>) -> Result<Json<Value>, StatusCode> {
    Ok(Json(json!({
        "status": "ok",
        "objects": [],
        "engine": "hailo-8-yolov8"
    })))
}
