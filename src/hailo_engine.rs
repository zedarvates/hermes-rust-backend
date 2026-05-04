use std::sync::OnceLock;

static AVAILABLE: OnceLock<bool> = OnceLock::new();

/// Check if Hailo-8 device is available
pub fn is_available() -> bool {
    *AVAILABLE.get_or_init(|| {
        std::path::Path::new("/dev/hailo0").exists()
    })
}

/// Get Hailo-8 device info
pub fn device_info() -> serde_json::Value {
    serde_json::json!({
        "device": "/dev/hailo0",
        "available": is_available(),
        "tops": 26,
        "architecture": "Hailo-8"
    })
}
