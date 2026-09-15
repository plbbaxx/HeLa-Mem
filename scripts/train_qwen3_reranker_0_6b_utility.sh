#!/usr/bin/env bash
set -euo pipefail

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

python -m hela_mem.train_qwen3_reranker_utility \
  --dataset-dir "${UTILITY_DATASET_DIR:-artifacts/utility_dataset_v1_1}" \
  --model-path "${QWEN3_RERANKER_MODEL_PATH:-/mnt/disk2/caoxue/models/Qwen3-Reranker-0.6B}" \
  --output-dir "${QWEN3_RERANKER_OUTPUT_DIR:-artifacts/qwen3_reranker_0_6b_utility_lora_v1}" \
  --historical-4b-dir "${MEMRERANKER_4B_OUTPUT_DIR:-artifacts/memreranker_utility_lora_v1}" \
  --epochs "${QWEN3_RERANKER_EPOCHS:-5}" \
  --learning-rate "${QWEN3_RERANKER_LR:-1e-5}" \
  --train-batch-size "${QWEN3_RERANKER_TRAIN_BATCH_SIZE:-0}" \
  --eval-batch-size "${QWEN3_RERANKER_EVAL_BATCH_SIZE:-4}" \
  --gradient-accumulation-steps "${QWEN3_RERANKER_GRAD_ACCUM:-1}" \
  --max-length "${QWEN3_RERANKER_MAX_LENGTH:-auto}" \
  --lora-r 16 \
  --lora-alpha 32 \
  --lora-dropout 0.05 \
  --attention "${QWEN3_RERANKER_ATTENTION:-sdpa}" \
  "$@"
