#!/usr/bin/env bash
set -euo pipefail
python -m hela_mem.build_utility_dataset \
 --data-path "${DATA_PATH:-data/longmemeval_s.json}" \
 --mem-dir "${MEM_DIR:-artifacts/longmemeval_qwen3_4b/run_500/encoded}" \
 --base-predictions "${BASE_PREDICTIONS:-artifacts/longmemeval_qwen3_4b/run_500/predictions.jsonl}" \
 --oracle-v05 "${ORACLE_V05:-artifacts/exact_oracle_edge_utility_v05/retrieval_scan.json}" \
 --local-model-path "${LOCAL_MODEL_PATH:-/mnt/disk2/caoxue/models/Qwen3-4B-Instruct-2507}" \
 --output-dir "${OUTPUT_DIR:-artifacts/utility_dataset_v1}" \
 --top-k "${TOP_K:-15}" --workers "${WORKERS:-8}" "$@"
