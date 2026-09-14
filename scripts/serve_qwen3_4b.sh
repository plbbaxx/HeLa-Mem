#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${QWEN_MODEL_PATH:-/mnt/disk2/caoxue/models/Qwen3-4B-Instruct-2507}"
SERVED_NAME="${QWEN_SERVED_NAME:-Qwen3-4B-Instruct-2507}"
PORT="${VLLM_PORT:-8000}"
MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-32768}"
GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.85}"

exec vllm serve "$MODEL_PATH" \
  --served-model-name "$SERVED_NAME" \
  --host 127.0.0.1 \
  --port "$PORT" \
  --dtype auto \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --tensor-parallel-size 1
