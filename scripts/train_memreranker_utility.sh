#!/usr/bin/env bash
set -euo pipefail

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

python -m hela_mem.train_memreranker_utility \
  --dataset-dir "${UTILITY_DATASET_DIR:-artifacts/utility_dataset_v1_1}" \
  --model-path "${MEMRERANKER_MODEL_PATH:-/mnt/disk2/caoxue/models/MemReranker-4B}" \
  --output-dir "${MEMRERANKER_OUTPUT_DIR:-artifacts/memreranker_utility_lora_v1}" \
  --max-length "${MEMRERANKER_MAX_LENGTH:-16384}" \
  --train-batch-size "${MEMRERANKER_TRAIN_BATCH_SIZE:-1}" \
  --eval-batch-size "${MEMRERANKER_EVAL_BATCH_SIZE:-1}" \
  --gradient-accumulation-steps "${MEMRERANKER_GRAD_ACCUM:-8}" \
  --epochs "${MEMRERANKER_EPOCHS:-5}" \
  --learning-rate "${MEMRERANKER_LR:-1e-5}" \
  --lora-r 16 \
  --lora-alpha 32 \
  --expected-train-questions 41 \
  "$@"
