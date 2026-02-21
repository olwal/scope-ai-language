#!/bin/sh
# Setup script for Daydream Scope Ollama pod on RunPod
# Usage: curl -fsSL https://raw.githubusercontent.com/olwal/scope-ai-language/main/scripts/setup-ollama-pod.sh | sh
#
# Installs Ollama, pulls the default VLM model, and starts the server
# bound to all interfaces (required for RunPod TCP port forwarding).
#
# Environment variables:
#   OLLAMA_MODEL  - model to pull (default: qwen3-vl:2b)

set -e

OLLAMA_MODEL="${OLLAMA_MODEL:-qwen3-vl:2b}"

echo "[scope] Installing dependencies..."
apt-get update -qq && apt-get install -y -q zstd lshw

echo "[scope] Installing Ollama..."
curl -fsSL https://ollama.com/install.sh | sh

echo "[scope] Pulling model: $OLLAMA_MODEL"
ollama pull "$OLLAMA_MODEL"

echo "[scope] Starting Ollama (binding to 0.0.0.0:11434)..."
exec env OLLAMA_HOST=0.0.0.0 ollama serve
