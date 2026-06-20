use std::sync::OnceLock;

static GPU_COUNT: OnceLock<usize> = OnceLock::new();
static GPU_INFO: OnceLock<Vec<GpuDevice>> = OnceLock::new();

#[derive(Debug, Clone, serde::Serialize)]
pub struct GpuDevice {
    pub index: usize,
    pub name: String,
    pub memory_gb: f32,
}

/// Check if CUDA GPUs are available
pub fn is_available() -> bool {
    device_count() > 0
}

/// Get number of CUDA devices
pub fn device_count() -> usize {
    *GPU_COUNT.get_or_init(|| {
        match std::process::Command::new("nvidia-smi")
            .args(["--query-gpu=name", "--format=csv,noheader"])
            .output()
        {
            Ok(output) => {
                let count = String::from_utf8_lossy(&output.stdout)
                    .lines()
                    .filter(|l| !l.is_empty())
                    .count();
                tracing::info!("🖥️ Detected {} CUDA GPU(s)", count);
                count
            }
            Err(e) => {
                tracing::warn!("nvidia-smi not available: {}", e);
                0
            }
        }
    })
}

/// Get detailed info about each GPU
pub fn gpu_info() -> &'static Vec<GpuDevice> {
    GPU_INFO.get_or_init(|| {
        let count = device_count();
        let mut devices = Vec::with_capacity(count);
        for i in 0..count {
            let name = std::process::Command::new("nvidia-smi")
                .args(["-i", &i.to_string(), "--query-gpu=name,memory.total",
                       "--format=csv,noheader,nounits"])
                .output()
                .ok()
                .and_then(|o| {
                    let line = String::from_utf8_lossy(&o.stdout).trim().to_string();
                    let parts: Vec<&str> = line.split(", ").collect();
                    if parts.len() == 2 {
                        Some(GpuDevice {
                            index: i,
                            name: parts[0].to_string(),
                            memory_gb: parts[1].parse::<f32>().unwrap_or(0.0) / 1024.0,
                        })
                    } else {
                        None
                    }
                })
                .unwrap_or(GpuDevice {
                    index: i,
                    name: "Unknown".into(),
                    memory_gb: 0.0,
                });
            devices.push(name);
        }
        devices
    })
}

/// Returns which GPU has the most free memory (for model loading)
pub fn best_gpu() -> usize {
    let devices = gpu_info();
    if devices.is_empty() { return 0; }
    // Simple: return the first GPU with most memory
    devices.iter()
        .max_by(|a, b| a.memory_gb.partial_cmp(&b.memory_gb).unwrap_or(std::cmp::Ordering::Equal))
        .map(|d| d.index)
        .unwrap_or(0)
}
