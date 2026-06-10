//! Hailo-8 NPU Engine
//!
//! Dual-mode: native HailoRT (when libhailort available) or REST bridge
//! to existing Python MCP server on EUREKAI:8767.
//!
//! Capabilities: classification (ResNet-18), detection (YOLOv8), OCR.
//! 26 TOPS, ~3-5W power draw.

use serde::{Deserialize, Serialize};
use std::sync::OnceLock;

static AVAILABLE: OnceLock<bool> = OnceLock::new();
static MCP_URL: OnceLock<String> = OnceLock::new();

// ── Public API ──────────────────────────────────────────────

/// Check if Hailo-8 device is available (local or remote)
pub fn is_available() -> bool {
    *AVAILABLE.get_or_init(|| {
        std::path::Path::new("/dev/hailo0").exists()
    })
}

/// Set MCP bridge URL (e.g., "http://192.168.1.47:8767")
pub fn set_mcp_url(url: &str) {
    let _ = MCP_URL.set(url.to_string());
}

fn mcp_url() -> &str {
    MCP_URL.get().map(|s| s.as_str()).unwrap_or("http://192.168.1.47:8767")
}

/// Get Hailo-8 device info
pub fn device_info() -> serde_json::Value {
    serde_json::json!({
        "device": "/dev/hailo0",
        "available": is_available(),
        "tops": 26,
        "architecture": "Hailo-8",
        "mcp_bridge": mcp_url(),
    })
}

// ── Classification ──────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassPrediction {
    pub label: String,
    pub confidence: f32,
}

/// Classify an image. Falls back to MCP bridge if no local Hailo.
pub async fn classify(image_path: &str) -> Result<Vec<ClassPrediction>, String> {
    if is_available() {
        classify_native(image_path).await
    } else {
        classify_via_mcp(image_path).await
    }
}

async fn classify_native(_image_path: &str) -> Result<Vec<ClassPrediction>, String> {
    Err("Native HailoRT classification not yet implemented. Use MCP bridge.".into())
}

async fn classify_via_mcp(image_path: &str) -> Result<Vec<ClassPrediction>, String> {
    let client = reqwest::Client::new();
    let resp = client
        .post(format!("{}/classify", mcp_url()))
        .json(&serde_json::json!({"image_path": image_path}))
        .send()
        .await
        .map_err(|e| format!("MCP request failed: {}", e))?;

    if !resp.status().is_success() {
        return Err(format!("MCP returned {}", resp.status()));
    }

    let body: serde_json::Value = resp.json().await.map_err(|e| e.to_string())?;

    // Parse response — adapt to actual MCP format
    let predictions: Vec<ClassPrediction> = body
        .get("predictions")
        .or_else(|| body.get("result"))
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|p| {
                    Some(ClassPrediction {
                        label: p.get("label")?.as_str()?.to_string(),
                        confidence: p.get("confidence")?.as_f64()? as f32,
                    })
                })
                .collect()
        })
        .unwrap_or_default();

    Ok(predictions)
}

// ── Detection ───────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Detection {
    pub label: String,
    pub confidence: f32,
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}

/// Detect objects in an image via Hailo-8 YOLOv8
pub async fn detect(image_path: &str) -> Result<Vec<Detection>, String> {
    let client = reqwest::Client::new();
    let resp = client
        .post(format!("{}/detect", mcp_url()))
        .json(&serde_json::json!({"image_path": image_path}))
        .send()
        .await
        .map_err(|e| format!("MCP request failed: {}", e))?;

    if !resp.status().is_success() {
        return Err(format!("MCP returned {}", resp.status()));
    }

    let body: serde_json::Value = resp.json().await.map_err(|e| e.to_string())?;

    let detections: Vec<Detection> = body
        .get("detections")
        .or_else(|| body.get("result"))
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|d| {
                    Some(Detection {
                        label: d.get("label")?.as_str()?.to_string(),
                        confidence: d.get("confidence")?.as_f64()? as f32,
                        x: d.get("x")?.as_f64()? as f32,
                        y: d.get("y")?.as_f64()? as f32,
                        width: d.get("width")?.as_f64()? as f32,
                        height: d.get("height")?.as_f64()? as f32,
                    })
                })
                .collect()
        })
        .unwrap_or_default();

    Ok(detections)
}

// ── OCR ─────────────────────────────────────────────────────

/// Extract text from an image via Hailo-8 OCR
pub async fn ocr(image_path: &str) -> Result<String, String> {
    let client = reqwest::Client::new();
    let resp = client
        .post(format!("{}/ocr", mcp_url()))
        .json(&serde_json::json!({"image_path": image_path}))
        .send()
        .await
        .map_err(|e| format!("MCP request failed: {}", e))?;

    if !resp.status().is_success() {
        return Err(format!("MCP returned {}", resp.status()));
    }

    let body: serde_json::Value = resp.json().await.map_err(|e| e.to_string())?;

    body
        .get("text")
        .or_else(|| body.get("result"))
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
        .ok_or_else(|| "No text in response".to_string())
}

// ── Benchmark ───────────────────────────────────────────────

#[derive(Debug, Clone, Serialize)]
pub struct HailoBenchmark {
    pub device: String,
    pub tops_rated: f32,
    pub inference_time_ms: u64,
    pub throughput_fps: f32,
}

/// Run a quick benchmark on the Hailo device
pub async fn benchmark() -> Result<HailoBenchmark, String> {
    if !is_available() {
        return Ok(HailoBenchmark {
            device: "MCP bridge (remote)".into(),
            tops_rated: 26.0,
            inference_time_ms: 0,
            throughput_fps: 0.0,
        });
    }

    // TODO: native benchmark
    Ok(HailoBenchmark {
        device: "/dev/hailo0".into(),
        tops_rated: 26.0,
        inference_time_ms: 0,
        throughput_fps: 0.0,
    })
}
