# Hermes Rust Backend 🦀⚡

[![Rust](https://img.shields.io/badge/rust-2021-orange.svg)]()
[![License](https://img.shields.io/badge/license-MIT-green.svg)]()
[![Version](https://img.shields.io/badge/version-0.3.0-blue.svg)]()

**High-performance Rust backend for Hermes Agent** — LocalAI proxy, MCP bridge, Hailo-8 vision, 3-GPU support.

> **v0.3.0** — Refonte complète : proxy LocalAI (plus de dépendance `llama-cpp-2`), bridge MCP (5 outils), 
> détection multi-GPU, streaming SSE. Déployé sur EUREKAI:8769.

## Quick Start

```bash
# Build
cargo build --release

# Run
./target/release/hermes-rust-backend
# → Listening on 0.0.0.0:8769
# → Auto-connecte à LocalAI (192.168.1.47:8080)
# → 8 modèles disponibles
```

## Architecture

```
hermes-rust-backend/
├── src/
│   ├── main.rs           — Serveur Axum, auto-découverte LocalAI
│   ├── gguf_engine.rs    — Proxy LocalAI (8 modèles, inférence réelle)
│   ├── openai_api.rs     — /v1/chat/completions, /v1/models (streaming)
│   ├── api_server.rs     — /status, endpoints Hermes
│   ├── mcp_bridge.rs     — 5 outils MCP pour Hermes Agent
│   ├── hailo_engine.rs   — Hailo-8 : classify/detect/OCR via MCP bridge
│   ├── multi_gpu.rs      — Détection 3 GPUs (2×RTX3060 + GTX1060)
│   ├── auth.rs           — JWT authentication
│   └── onnx_engine.rs    — ONNX Runtime (détection automatique)
└── Cargo.toml
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Status + version + features |
| `/engines` | GET | Tous les engines (GGUF, Hailo, CUDA, ONNX) |
| `/v1/chat/completions` | POST | OpenAI-compatible (proxy LocalAI) |
| `/v1/models` | GET | Modèles disponibles depuis LocalAI |
| `/status` | GET | Status engines + GPU count |
| `/mcp/tools` | GET | 5 outils MCP pour Hermes Agent |

### MCP Tools (Hermes Integration)

```
mcp_gguf_infer     — Générer du texte via LocalAI
mcp_hailo_classify — Classifier une image via Hailo-8 ResNet
mcp_hailo_detect   — Détection YOLOv8m via Hailo-8
mcp_hailo_ocr      — OCR via Hailo-8 Tesseract
mcp_engine_status  — Statut de tous les engines
```

## Déploiement (EUREKAI)

```bash
# Sur EUREKAI (192.168.1.47):
cd ~/hermes-rust-backend
git pull
cargo build --release
pkill -f hermes-rust-backend
nohup ./target/release/hermes-rust-backend > /tmp/rust_backend.log 2>&1 &

# Vérifier
curl localhost:8769/health
curl localhost:8769/engines
curl localhost:8769/mcp/tools
```

## Utilisation depuis Hermes

```bash
# Chat via proxy LocalAI
curl -X POST localhost:8769/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"gpt-4o","messages":[{"role":"user","content":"Hello"}]}'

# MCP tools
curl localhost:8769/mcp/tools
```

## Performance

- **LocalAI Proxy** — 8 modèles (gpt-4o, whisper-1, tts-1, stable-diffusion, etc.)
- **Hailo-8** — 26 TOPS NPU, REST bridge (192.168.1.47:8767)
- **3 GPUs** — 2× RTX 3060 + GTX 1060
- **Timing** — Chaque inference retourne tokens/sec + ms

## Dépendances

- Rust 2021 edition
- LocalAI sur EUREKAI (:8080) — requis pour l'inférence
- Hailo-8 API (:8767) — optionnel
- Aucune dépendance C (plus de `llama-cpp-2` / `libclang`)

## Projets liés

- [cogniarc](https://github.com/zedarvates/cogniarc) — Cognitive reasoning engine
- [hyperframes](https://github.com/heygen-com/hyperframes) — HTML-to-Video framework
- [hermes-brain](https://github.com/zedarvates/hermes-brain) — Architecture cognitive

## Licence MIT


---

[![Donate](https://img.shields.io/badge/☕%20Soutenir-BTC%20%7C%20ETH-orange)](DONATE.md)