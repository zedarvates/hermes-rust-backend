//! MCP (Model Context Protocol) bridge for Hermes Agent integration
//! Exposes Rust backend tools as MCP tools that Hermes can discover and call.

use axum::{Router, Json, routing::get, extract::State};
use serde_json::{json, Value};
use std::sync::Arc;

pub fn mcp_routes() -> Router<Arc<super::openai_api::AppState>> {
    Router::new()
        .route("/mcp/tools", get(list_tools))
        .route("/mcp/tools/:name", get(tool_info))
}

#[derive(Debug, serde::Serialize)]
struct McpTool {
    name: String,
    description: String,
    input_schema: Value,
}

async fn list_tools() -> Json<Value> {
    Json(json!({
        "tools": [
            {
                "name": "mcp_gguf_infer",
                "description": "Generate text using the loaded GGUF model",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "prompt": {"type": "string", "description": "Input prompt"},
                        "max_tokens": {"type": "integer", "default": 256}
                    },
                    "required": ["prompt"]
                }
            },
            {
                "name": "mcp_hailo_classify",
                "description": "Classify an image using Hailo-8 ResNet-18",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "image_path": {"type": "string", "description": "Path to image file"}
                    },
                    "required": ["image_path"]
                }
            },
            {
                "name": "mcp_hailo_detect",
                "description": "Detect objects in an image using Hailo-8 YOLOv8",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "image_path": {"type": "string", "description": "Path to image file"}
                    },
                    "required": ["image_path"]
                }
            },
            {
                "name": "mcp_hailo_ocr",
                "description": "Extract text from an image using Hailo-8 OCR",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "image_path": {"type": "string", "description": "Path to image file"}
                    },
                    "required": ["image_path"]
                }
            },
            {
                "name": "mcp_engine_status",
                "description": "Get status of all AI engines (GGUF, Hailo-8, CUDA)",
                "inputSchema": {
                    "type": "object",
                    "properties": {}
                }
            }
        ]
    }))
}

async fn tool_info() -> Json<Value> {
    Json(json!({
        "description": "Hermes Rust Backend MCP Bridge",
        "version": "0.3.0",
        "tools_count": 5,
        "endpoint": "/mcp/tools"
    }))
}
