use serde::{Deserialize, Serialize};
use std::sync::OnceLock;

static LOCALAI_URL: OnceLock<String> = OnceLock::new();

pub fn set_localai_url(url: &str) {
    let _ = LOCALAI_URL.set(url.to_string());
}

fn localai_url() -> &str {
    LOCALAI_URL.get().map(|s| s.as_str()).unwrap_or("http://192.168.1.47:8080")
}

#[derive(Debug, Clone, PartialEq)]
pub enum GpuBackend { CUDA, Vulkan, CPU }

impl std::fmt::Display for GpuBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GpuBackend::CUDA => write!(f, "CUDA"),
            GpuBackend::Vulkan => write!(f, "Vulkan"),
            GpuBackend::CPU => write!(f, "CPU"),
        }
    }
}

pub fn detect_gpu_backend() -> (GpuBackend, u32) {
    let cuda = std::process::Command::new("nvidia-smi")
        .output().map(|o| o.status.success()).unwrap_or(false);
    if cuda {
        let count = std::process::Command::new("nvidia-smi")
            .args(["--query-gpu=name", "--format=csv,noheader"])
            .output().ok()
            .map(|o| String::from_utf8_lossy(&o.stdout).lines()
                .filter(|l| !l.is_empty()).count() as u32)
            .unwrap_or(1);
        return (GpuBackend::CUDA, count);
    }
    (GpuBackend::CPU, 0)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceResult {
    pub text: String,
    pub tokens_generated: u32,
    pub tokens_per_second: f64,
    pub total_time_ms: u64,
}

pub struct GgufEngine {
    pub backend: GpuBackend,
    pub gpu_count: u32,
    pub model_loaded: bool,
    models: Vec<String>,
}

impl GgufEngine {
    pub fn new() -> Self {
        let (backend, gpu_count) = detect_gpu_backend();
        Self { backend, gpu_count, model_loaded: false, models: vec![] }
    }

    pub fn is_available(&self) -> bool {
        self.model_loaded
    }

    pub fn list_models(&self) -> &[String] {
        &self.models
    }

    pub async fn refresh_models(&mut self) -> Result<(), String> {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(5))
            .build().map_err(|e| e.to_string())?;
        let resp = client.get(format!("{}/v1/models", localai_url()))
            .send().await.map_err(|e| format!("LocalAI unreachable at {}: {}", localai_url(), e))?;
        let body: serde_json::Value = resp.json().await.map_err(|e| e.to_string())?;
        self.models = body["data"].as_array()
            .map(|arr| arr.iter()
                .filter_map(|m| m["id"].as_str().map(String::from))
                .collect())
            .unwrap_or_default();
        self.model_loaded = !self.models.is_empty();
        Ok(())
    }

    pub async fn infer(&self, prompt: &str, max_tokens: u32) -> Result<InferenceResult, String> {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(60))
            .build().map_err(|e| e.to_string())?;
        let model = self.models.first().cloned().unwrap_or_else(|| "gpt-4o".into());
        let start = std::time::Instant::now();
        let resp = client
            .post(format!("{}/v1/chat/completions", localai_url()))
            .json(&serde_json::json!({
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": 0.7
            }))
            .send().await.map_err(|e| format!("LocalAI request failed: {}", e))?;
        let elapsed = start.elapsed();
        let body: serde_json::Value = resp.json().await.map_err(|e| e.to_string())?;
        let text = body["choices"][0]["message"]["content"].as_str().unwrap_or("").to_string();
        let tokens = body["usage"]["completion_tokens"].as_u64().unwrap_or(0) as u32;
        Ok(InferenceResult {
            text,
            tokens_generated: tokens,
            tokens_per_second: tokens as f64 / elapsed.as_secs_f64().max(0.001),
            total_time_ms: elapsed.as_millis() as u64,
        })
    }

    pub async fn infer_default(&self, prompt: &str) -> Result<InferenceResult, String> {
        self.infer(prompt, 512).await
    }
}
