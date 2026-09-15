#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR="${MEMRERANKER_OUTPUT_DIR:-artifacts/memreranker_utility_lora_v1}"
LOG_PATH="${MEMRERANKER_LOG_PATH:-${OUTPUT_DIR}/run.log}"

echo "Processes:"
pgrep -af "hela_mem.train_memreranker_utility" || true
echo
echo "GPU:"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader || true
echo
echo "Latest log: ${LOG_PATH}"
tail -n 30 "${LOG_PATH}" 2>/dev/null || true
echo
echo "Completed artifacts:"
for path in \
  "${OUTPUT_DIR}/input_length_audit.json" \
  "${OUTPUT_DIR}/baseline_results.json" \
  "${OUTPUT_DIR}/training_history.json" \
  "${OUTPUT_DIR}/training_summary.json" \
  "${OUTPUT_DIR}/best_adapter/selection.json" \
  "${OUTPUT_DIR}/final_test_results.json"; do
  if [[ -f "${path}" ]]; then echo "OK  ${path}"; else echo "--  ${path}"; fi
done
