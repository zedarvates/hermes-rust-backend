# 🦀 Hermes Rust Backend

> **High-performance inference backend for [Hermes Agent](https://github.com/nousresearch/hermes-agent)**
>
> Built in Rust. Runs on your own hardware. Zero tokens wasted on cloud APIs for local inference.

[![Rust](https://img.shields.io/badge/Rust-1.95%2B-orange?logo=rust)](https://www.rust-lang.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Axum](https://img.shields.io/badge/Axum-0.7-blue?logo=rust)](https://github.com/tokio-rs/axum)
[![ONNX Runtime](https://img.shields.io/badge/ONNX%20Runtime-2.0-4B0082)](https://github.com/microsoft/onnxruntime)

---

## Why This Exists

Hermes Agent is powerful, but when it comes to **local model inference**, every call to a cloud API costs tokens and leaks data. This backend gives you:

- **100% local inference** — no API keys, no cloud, complete privacy
- **Multi-engine support** — ONNX Runtime, Candle, Hailo-8 NPU
- **Multi-GPU distribution** — load balance across 2+ GPUs
- **REST API** — drop-in replacement for cloud inference endpoints
- **Hermes MCP bridge** — native integration with Hermes Agent

If you already have GPUs (or a Hailo-8 NPU) sitting idle, this backend turns them into your own private inference cloud.

---

## Engine Support

| Engine | Hardware | Models | Use Case |
|--------|----------|--------|----------|
| **ONNX Runtime** | NVIDIA GPU, CPU | SenseNova-U1, YOLOv8, ResNet | Image generation, object detection, classification |
| **Candle** | NVIDIA GPU, CPU, M1/M2 | Gemma 4, Mistral, Llama | LLM inference (fallback) |
| **Hailo-8** | Hailo-8 NPU (PCIe/M.2) | YOLOv8m, ResNet-18, PaddleOCR | Ultra-efficient edge vision (26 TOPS @ ~2.5W) |
| **Multi-GPU** | 2+ NVIDIA GPUs | Any ONNX/Candle model | Distributed inference, load balancing |

---

## Quick Start

### Prerequisites

```bash
# Rust toolchain
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# CUDA (for NVIDIA GPU inference)
# Install CUDA 12.x from https://developer.nvidia.com/cuda-downloads

# Hailo-8 (optional, for edge vision)
# Install hailo-pci-driver and HailoRT from https://hailo.ai/developer-zone/
```

### Install & Run

```bash
# Clone
git clone https://github.com/zedarvates/hermes-rust-backend.git
cd hermes-rust-backend

# Build (debug)
cargo build

# Run
cargo run
# 🚀 Server listening on http://0.0.0.0:8769
```

### Release Build (optimized)

```bash
cargo build --release
# Binary: target/release/hermes-rust-backend (~5 MB)
./target/release/hermes-rust-backend
```

---

## API Reference

### `GET /health`

```json
{"status": "ok", "version": "0.1.0"}
```

### `GET /v1/status`

```json
{
  "status": "healthy",
  "engines": {
    "onnx": true,
    "hailo": true,
    "cuda_devices": 2
  },
  "version": "0.1.0"
}
```

### `GET /engines`

```json
{
  "engines": [
    {"name": "onnx", "available": true, "devices": ["cuda:0", "cuda:1"]},
    {"name": "candle", "available": true, "devices": ["cuda:0"]},
    {"name": "hailo", "available": true, "device": "/dev/hailo0"}
  ]
}
```

### `POST /v1/infer`

Inference endpoint (ONNX Runtime).

```bash
curl -X POST http://192.168.1.47:8769/v1/infer \
  -H "Content-Type: application/json" \
  -d '{"engine": "onnx", "model": "yolo_v8m", "image": "<base64>"}'
```

### `POST /v1/ocr`

OCR endpoint (Hailo-8 PaddleOCR or Tesseract fallback).

```bash
curl -X POST http://192.168.1.47:8769/v1/ocr \
  -H "Content-Type: application/json" \
  -d '{"image": "<base64>", "engine": "hailo"}'
```

### `POST /v1/detect`

Object detection endpoint.

```bash
curl -X POST http://192.168.1.47:8769/v1/detect \
  -H "Content-Type: application/json" \
  -d '{"image": "<base64>", "model": "yolo_v8m"}'
```

---

## Project Structure

```
hermes-rust-backend/
├── Cargo.toml              # Dependencies (ort, candle, axum, cudarc...)
└── src/
    ├── main.rs             # Entrypoint, server startup, route registration
    ├── lib.rs              # Library root, module declarations
    ├── api_server.rs       # Axum HTTP server, REST endpoints
    ├── auth.rs             # JWT authentication, rate limiting
    ├── onnx_engine.rs      # ONNX Runtime engine, model loading/inference
    ├── candle_engine.rs    # Candle engine (HuggingFace models, fallback)
    ├── hailo_engine.rs     # Hailo-8 NPU engine (26 TOPS, edge vision)
    └── multi_gpu.rs        # Multi-GPU distribution, load balancing
```

---

## Deployment

### Systemd Service

```bash
sudo cp hermes-rust-backend.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now hermes-rust-backend
```

Example service file:

```ini
[Unit]
Description=Hermes Rust Backend
After=network.target

[Service]
Type=simple
User=sylvain
WorkingDirectory=/home/sylvain/hermes-rust-backend
ExecStart=/home/sylvain/hermes-rust-backend/target/release/hermes-rust-backend
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

### Docker (Coming Soon)

```bash
docker build -t hermes-rust-backend .
docker run --gpus all -p 8769:8769 hermes-rust-backend
```

---

## Cluster Setup (Real-World Example)

This backend runs on a **3-machine cluster**:

```
┌──────────────── ─┐   ┌─────────────────┐   ┌─────────────────┐
│ GLYPH  mini-pc   │   │    EUREKAI      │   │  THOR laptop    │
│  Hermes Agent    │──►│  Rust Backend   │   │  Home Assistant │
│  (orchestrator)  │   │  2x RTX 3060    │   │  Automations    │
│  :3001           │   │  Hailo-8 NPU    │   │  :8123          │
│                  │   │  :8769          │   │                 │
└──────────────── ─┘   └─────────────────┘   └─────────────────┘
         │                      │
         │    ┌─────────────────┘
         ▼    ▼
    Inference requests distributed across engines:
    • ONNX Runtime → SenseNova-U1 (image gen, infographics)
    • Candle → Gemma 4 (LLM fallback)
    • Hailo-8 → YOLOv8m (72 FPS, object detection), PaddleOCR
```

---

## Dependencies

| Crate | Version | Purpose |
|-------|---------|---------|
| [`ort`](https://github.com/pykeio/ort) | 2.0-rc12 | ONNX Runtime bindings |
| [`axum`](https://github.com/tokio-rs/axum) | 0.7 | HTTP server |
| [`tokio`](https://github.com/tokio-rs/tokio) | 1.52 | Async runtime |
| [`cudarc`](https://github.com/coreylowman/cudarc) | 0.12 | CUDA driver API |
| [`image`](https://github.com/image-rs/image) | 0.25 | Image processing |
| [`reqwest`](https://github.com/seanmonstar/reqwest) | 0.12 | HTTP client |
| [`jsonwebtoken`](https://github.com/Keats/jsonwebtoken) | 9 | JWT auth |
| [`serde`](https://github.com/serde-rs/serde) | 1 | Serialization |
| [`tracing`](https://github.com/tokio-rs/tracing) | 0.1 | Structured logging |

Total: **322 crates**, compiled in < 20s (incremental) / < 2 min (from scratch).

---

## Contributing

Contributions are welcome! This backend is designed to be the community default for local Hermes Agent inference.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amd-rocm-support`)
3. Commit your changes
4. Push and open a Pull Request

**Areas for contribution:**
- AMD ROCm / Apple Metal GPU support
- More model engines (GGUF via llama.cpp bindings, vLLM)
- WebSocket streaming inference
- Docker image + Kubernetes Helm chart
- GPU monitoring dashboard
- Benchmarking suite

---

## License

MIT — see [LICENSE](LICENSE) for details.

---

## Related Projects

- [Hermes Agent](https://github.com/nousresearch/hermes-agent) — The agent that uses this backend
- [SenseNova-U1](https://github.com/OpenSenseNova/SenseNova-U1) — Unified text+image generation model
- [SenseNova-Skills](https://github.com/OpenSenseNova/SenseNova-Skills) — 23 skills for Hermes Agent
- [HailoRT](https://github.com/hailo-ai/hailort) — Hailo-8 runtime (MIT licensed)
- [Ort](https://github.com/pykeio/ort) — Rust ONNX Runtime bindings
- [Candle](https://github.com/huggingface/candle) — HuggingFace Rust ML framework

---

*Built with 🦀 for the Hermes community.*
