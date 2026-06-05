# Hermes Rust Backend 🦀⚡

[![Rust](https://img.shields.io/badge/rust-2024-orange.svg)]()
[![License](https://img.shields.io/badge/license-MIT-green.svg)]()
[![Status](https://img.shields.io/badge/status-active-brightgreen.svg)]()

High-performance Rust backend for Hermes Agent — ONNX Runtime, Candle, Hailo-8, multi-GPU, REST API.

## Architecture

```
hermes-rust-backend/
├── src/
│   ├── inference/    — Moteur d'inférence (ONNX, Candle)
│   ├── hailo/        — Hailo-8 NPU acceleration
│   ├── api/          — REST API endpoints
│   └── config/       — Configuration
├── models/           — Modèles optimisés
└── benchmarks/       — Tests de performance
```

## API Endpoints

| Endpoint | Méthode | Description |
|----------|---------|-------------|
| `/v1/infer` | POST | Inférence générique |
| `/v1/models` | GET | Liste des modèles chargés |
| `/v1/health` | GET | Statut du backend |
| `/v1/stats` | GET | Métriques de performance |

## Performance

- **ONNX Runtime** — Inférence CPU/GPU optimisée
- **Candle** — Framework ML Rust natif (pas de Python)
- **Hailo-8** — 26 TOPS NPU pour vision/traitement
- **Multi-GPU** — Répartition de charge entre GPU disponibles

## Projets liés

- [hermes-brain](https://github.com/zedarvates/hermes-brain) — Architecture cognitive
- [kitten-tts](https://github.com/zedarvates/kitten-tts) — TTS local FR
- [hnoss-voice](https://github.com/zedarvates/hnoss-voice) — Assistant vocal

## Licence MIT
