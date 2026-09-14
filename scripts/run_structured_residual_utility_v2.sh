#!/usr/bin/env bash
set -euo pipefail

DATA_PATH="data/longmemeval_s.json"
MEM_DIR="artifacts/longmemeval_qwen3_4b/run_500/encoded"
ORACLE_V05="artifacts/exact_oracle_edge_utility_v05/retrieval_scan.json"
V1_REPORT="artifacts/utility_scorer_v1/utility_scorer_v1_report.json"
OUTPUT_DIR="artifacts/structured_residual_utility_v2"
WORKERS=8
while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-path) DATA_PATH="$2"; shift 2 ;;
    --mem-dir) MEM_DIR="$2"; shift 2 ;;
    --oracle-v05) ORACLE_V05="$2"; shift 2 ;;
    --v1-report) V1_REPORT="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --workers) WORKERS="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

python -m hela_mem.structured_residual_utility_v2 \
  --data-path "$DATA_PATH" \
  --mem-dir "$MEM_DIR" \
  --oracle-v05 "$ORACLE_V05" \
  --v1-report "$V1_REPORT" \
  --output-dir "$OUTPUT_DIR" \
  --workers "$WORKERS" \
  --top-k 15
