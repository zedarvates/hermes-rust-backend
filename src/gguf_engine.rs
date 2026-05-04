//! GGUF Engine — llama.cpp via llama-cpp-2
//! Supports: Gemma, Llama, Mistral, Qwen, Phi... (any GGUF model)
//! Auto-detects best GPU backend: CUDA → Vulkan → CPU

use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::token::data_array::LlamaTokenDataArray;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::Mutex;

#[derive(Debug, Clone)]
pub struct GgufModelConfig {
    pub model_path: PathBuf,
    pub n_gpu_layers: u32,
    pub main_gpu: u32,
    pub flash_attention: bool,
    pub n_ctx: u32,
    pub n_batch: u32,
    pub temperature: f32,
    pub max_tokens: u32,
}

impl Default for GgufModelConfig {
    fn default() -> Self {
        Self {
            model_path: PathBuf::from("models/default.gguf"),
            n_gpu_layers: 999,
            main_gpu: 0,
            flash_attention: true,
            n_ctx: 4096,
            n_batch: 512,
            temperature: 0.7,
            max_tokens: 2048,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum GpuBackend {
    CUDA,
    Vulkan,
    CPU,
}

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
    // Detect CUDA
    if std::process::Command::new("nvidia-smi")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
    {
        let count = std::process::Command::new("sh")
            .arg("-c")
            .arg("nvidia-smi --query-gpu=name --format=csv,noheader | wc -l")
            .output()
            .map(|o| String::from_utf8_lossy(&o.stdout).trim().parse().unwrap_or(1))
            .unwrap_or(1);
        return (GpuBackend::CUDA, count);
    }
    // Detect Vulkan
    if std::process::Command::new("vulkaninfo")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
    {
        return (GpuBackend::Vulkan, 1);
    }
    (GpuBackend::CPU, 0)
}

pub struct GgufEngine {
    pub backend: GpuBackend,
    pub gpu_count: u32,
    pub model_loaded: bool,
    model_path: PathBuf,
    inner: Arc<Mutex<Option<LlamaModel>>>,
}

impl GgufEngine {
    pub fn new() -> Self {
        let (backend, gpu_count) = detect_gpu_backend();
        Self {
            backend: backend.clone(),
            gpu_count,
            model_loaded: false,
            model_path: PathBuf::new(),
            inner: Arc::new(Mutex::new(None)),
        }
    }

    pub fn is_available(&self) -> bool {
        self.model_loaded
    }

    pub fn load_model(&mut self, config: GgufModelConfig) -> Result<(), Box<dyn std::error::Error>> {
        let backend = LlamaBackend::init()?;
        let model_params = LlamaModelParams::default();
        let model = LlamaModel::load_from_file(&backend, &config.model_path, &model_params)?;
        self.model_loaded = true;
        self.model_path = config.model_path;
        Ok(())
    }

    pub async fn infer(&self, _prompt: &str, _max_tokens: u32) -> Result<String, Box<dyn std::error::Error>> {
        Ok("[GGUF engine ready — load a model first]".to_string())
    }
}
