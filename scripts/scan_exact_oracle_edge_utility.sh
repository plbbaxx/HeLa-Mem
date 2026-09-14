#!/usr/bin/env bash
set -euo pipefail

DATA_PATH="data/longmemeval_s.json"
MEM_DIR="artifacts/longmemeval_qwen3_4b/run_500/encoded"
BASE_PREDICTIONS="artifacts/longmemeval_qwen3_4b/run_500/predictions.jsonl"
QUALITY_REPORT="artifacts/cross_edge_quality_diagnostic/quality.json"
OUTPUT_DIR="artifacts/exact_oracle_edge_utility_v05"
WORKERS=4
while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-path) DATA_PATH="$2"; shift 2 ;;
    --mem-dir) MEM_DIR="$2"; shift 2 ;;
    --base-predictions) BASE_PREDICTIONS="$2"; shift 2 ;;
    --quality-report) QUALITY_REPORT="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --workers) WORKERS="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

python -m hela_mem.scan_exact_oracle_edge_utility \
  --data-path "$DATA_PATH" \
  --mem-dir "$MEM_DIR" \
  --base-predictions "$BASE_PREDICTIONS" \
  --quality-report "$QUALITY_REPORT" \
  --output-dir "$OUTPUT_DIR" \
  --workers "$WORKERS" \
  --top-k 15 \
  --max-flipped 3 \
  --activation-alpha 0.1 \
  --spreading-threshold 0.4 \
  --keyword-weight 0.5
