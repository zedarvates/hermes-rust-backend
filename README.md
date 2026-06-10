# Hermes Rust Backend 🦀⚡

[![Rust](https://img.shields.io/badge/rust-2021-orange.svg)]()
[![License](https://img.shields.io/badge/license-MIT-green.svg)]()
[![Status](https://img.shields.io/badge/status-active-brightgreen.svg)]()

High-performance Rust backend for Hermes Agent — GGUF inference, Hailo-8 vision, OpenAI-compatible API.

## Architecture

```
hermes-rust-backend/
├── src/
│   ├── gguf_engine.rs     — llama.cpp GGUF inference (llama-cpp-2)
│   ├── hailo_engine.rs    — Hailo-8 NPU: classify/detect/OCR
│   ├── openai_api.rs      — /v1/chat/completions, /v1/models
│   ├── api_server.rs      — Hermes-specific endpoints
│   └── auth.rs            — JWT authentication
├── models/                — GGUF model files
└── Cargo.toml
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/chat/completions` | POST | OpenAI-compatible chat (real GGUF inference) |
| `/v1/models` | GET | Available models |
| `/health` | GET | Service status + model availability |
| `/engines` | GET | GGUF + Hailo-8 engine status |

## Build Requirements

- **Rust** 1.75+
- **libclang** (for llama-cpp-2 bindgen): `sudo apt install libclang-dev`
- **CUDA Toolkit** (optional): `nvidia-cuda-toolkit`
- Target: **EUREKAI** (Linux x86_64, GPU recommended)

```bash
cargo build --release
./target/release/hermes-rust-backend
# → Listening on 0.0.0.0:8769
```

## Quick Start

```bash
# Drop a GGUF model
mkdir -p models/
cp ~/models/gemma-4-e2b-it-Q4_K_M.gguf models/

# Start server
cargo run --release
```

Then use as OpenAI provider in Hermes:
```bash
hermes config set model.base_url http://192.168.1.47:8769/v1
hermes config set model.default gguf
```

## Performance

- **GGUF Engine** — llama.cpp native, token sampling with temperature
- **Hailo-8** — 26 TOPS NPU, REST bridge to Python MCP
- **Timing** — every inference returns tokens/sec + ms elapsed

## Projets liés

- [cogniarc](https://github.com/zedarvates/cogniarc) — Cognitive reasoning engine
- [hermes-brain](https://github.com/zedarvates/hermes-brain) — Architecture cognitive
- [kitten-tts](https://github.com/zedarvates/kitten-tts) — TTS local FR

## Licence MIT
