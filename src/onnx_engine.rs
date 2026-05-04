use std::sync::OnceLock;
static AVAILABLE: OnceLock<bool> = OnceLock::new();
pub fn is_available() -> bool {
    *AVAILABLE.get_or_init(|| {
        let paths = ["/usr/lib/libonnxruntime.so", "/usr/local/lib/libonnxruntime.so"];
        let ok = paths.iter().any(|p| std::path::Path::new(p).exists());
        if ok { tracing::info!("ONNX Runtime found"); }
        ok
    })
}
pub async fn load_sensenova(path: &str) -> anyhow::Result<()> {
    if !is_available() { anyhow::bail!("ONNX Runtime not found"); }
    tracing::info!("SenseNova path: {}", path);
    Ok(())
}
