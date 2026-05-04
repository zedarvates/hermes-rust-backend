use std::sync::OnceLock;

static GPU_COUNT: OnceLock<usize> = OnceLock::new();

/// Check if CUDA GPUs are available
pub fn is_available() -> bool {
    device_count() > 0
}

/// Get number of CUDA devices
pub fn device_count() -> usize {
    *GPU_COUNT.get_or_init(|| {
        // Try to detect CUDA devices
        match std::process::Command::new("nvidia-smi")
            .args(["--query-gpu=name", "--format=csv,noheader"])
            .output()
        {
            Ok(output) => {
                let count = String::from_utf8_lossy(&output.stdout)
                    .lines()
                    .filter(|l| !l.is_empty())
                    .count();
                tracing::info!("Detected {} CUDA GPU(s)", count);
                count
            }
            Err(e) => {
                tracing::warn!("nvidia-smi not available: {}", e);
                0
            }
        }
    })
}
