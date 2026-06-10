//! GGUF Engine — llama.cpp via llama-cpp-2
//! Supports: Gemma, Llama, Mistral, Qwen, Phi... (any GGUF model)
//! Auto-detects best GPU backend: CUDA → Vulkan → CPU

use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::token::data_array::LlamaTokenDataArray;
use llama_cpp_2::token::LlamaToken;
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
    if std::process::Command::new("vulkaninfo")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
    {
        return (GpuBackend::Vulkan, 1);
    }
    (GpuBackend::CPU, 0)
}

/// Full inference result with timing stats
#[derive(Debug, Clone)]
pub struct InferenceResult {
    pub text: String,
    pub tokens_generated: u32,
    pub tokens_per_second: f64,
    pub total_time_ms: u64,
}

struct LoadedModel {
    backend: LlamaBackend,
    model: LlamaModel,
    context: LlamaContext,
    config: GgufModelConfig,
}

pub struct GgufEngine {
    pub backend: GpuBackend,
    pub gpu_count: u32,
    pub model_loaded: bool,
    model_path: PathBuf,
    inner: Arc<Mutex<Option<LoadedModel>>>,
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

    pub fn load_model(
        &mut self,
        config: GgufModelConfig,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let backend = LlamaBackend::init()?;

        let model_params = LlamaModelParams::default()
            .with_n_gpu_layers(config.n_gpu_layers)
            .with_main_gpu(config.main_gpu);

        let model = LlamaModel::load_from_file(&backend, &config.model_path, &model_params)?;

        // Create context with flash attention if CUDA
        let mut ctx_params = LlamaContextParams::default()
            .with_n_ctx(Some(config.n_ctx))
            .with_n_batch(config.n_batch);

        if config.flash_attention && self.backend == GpuBackend::CUDA {
            ctx_params = ctx_params.with_flash_attention(true);
        }

        let context = model.new_context(&backend, ctx_params)?;

        let loaded = LoadedModel {
            backend,
            model,
            context,
            config: config.clone(),
        };

        let mut guard = self.inner.blocking_lock();
        *guard = Some(loaded);
        self.model_loaded = true;
        self.model_path = config.model_path;

        Ok(())
    }

    /// Run inference with the loaded GGUF model.
    /// Returns generated text with performance stats.
    pub async fn infer(
        &self,
        prompt: &str,
        max_tokens: u32,
    ) -> Result<InferenceResult, Box<dyn std::error::Error>> {
        let guard = self.inner.lock().await;
        let loaded = guard
            .as_ref()
            .ok_or("No model loaded. Call load_model() first.")?;

        let start = std::time::Instant::now();

        // Tokenize input
        let tokens = loaded
            .model
            .str_to_token(prompt, llama_cpp_2::token::data_array::LlamaTokenAttr::ALL_NORMAL)?;

        let n_tokens = tokens.len() as i32;
        let max_gen = max_tokens.min(loaded.config.max_tokens) as i32;

        // Initialize batch
        let mut batch = LlamaBatch::new(n_tokens as u32 + max_gen as u32, 1)?;

        // Add prompt tokens to batch
        for (i, token) in tokens.iter().enumerate() {
            batch.add(*token, i as i32, &[0], i == n_tokens as usize - 1)?;
        }

        // Decode the prompt
        loaded.context.decode(&mut batch)?;

        // Generate tokens
        let mut generated: Vec<LlamaToken> = Vec::with_capacity(max_gen as usize);
        let eos_token = loaded.model.token_eos();

        for pos in 0..max_gen {
            // Sample next token
            let candidates = loaded.context.candidates();
            let mut candidates_p = LlamaTokenDataArray::from_iter(candidates, false);

            // Apply temperature
            candidates_p.sample_temp(loaded.config.temperature);

            // Greedy sample (top-1)
            let next_token = candidates_p.sample_token();

            if next_token == eos_token {
                break;
            }

            generated.push(next_token);

            // Prepare next batch with single token
            batch.clear();
            batch.add(next_token, n_tokens + pos, &[0], true)?;
            loaded.context.decode(&mut batch)?;
        }

        let elapsed = start.elapsed();
        let tokens_gen = generated.len() as u32;
        let tps = if elapsed.as_secs_f64() > 0.0 {
            tokens_gen as f64 / elapsed.as_secs_f64()
        } else {
            0.0
        };

        // Detokenize output
        let text = loaded.model.tokens_to_str(&generated)?;

        Ok(InferenceResult {
            text,
            tokens_generated: tokens_gen,
            tokens_per_second: tps,
            total_time_ms: elapsed.as_millis() as u64,
        })
    }

    /// Infer with default max_tokens from config
    pub async fn infer_default(
        &self,
        prompt: &str,
    ) -> Result<InferenceResult, Box<dyn std::error::Error>> {
        let max_tok = {
            let guard = self.inner.lock().await;
            guard
                .as_ref()
                .map(|l| l.config.max_tokens)
                .unwrap_or(256)
        };
        self.infer(prompt, max_tok).await
    }
}
