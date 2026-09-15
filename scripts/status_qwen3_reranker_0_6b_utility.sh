#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR="${QWEN3_RERANKER_OUTPUT_DIR:-artifacts/qwen3_reranker_0_6b_utility_lora_v1}"
LOG_PATH="${QWEN3_RERANKER_LOG_PATH:-${OUTPUT_DIR}/run.log}"

pgrep -af "hela_mem.train_qwen3_reranker_utility" || true
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader || true
tail -n 30 "${LOG_PATH}" 2>/dev/null || true

for path in config.json token_length_stats.json baseline_metrics.json train_log.jsonl dev_metrics_by_epoch.json best_checkpoint/selection.json test_metrics.json runtime_stats.json cost_comparison.json; do
  if [[ -f "${OUTPUT_DIR}/${path}" ]]; then
    echo "OK  ${OUTPUT_DIR}/${path}"
  else
    echo "--  ${OUTPUT_DIR}/${path}"
  fi
done
