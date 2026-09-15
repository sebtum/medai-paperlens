#!/bin/sh
set -e

ollama serve &
OLLAMA_PID=$!

echo "Waiting for Ollama server..."
until ollama list > /dev/null 2>&1; do
  sleep 2
done

echo "Pulling model: ${OLLAMA_MODEL:-qwen3:0.6b}"
ollama pull "${OLLAMA_MODEL:-qwen3:0.6b}"
echo "Model ready."

wait "$OLLAMA_PID"
