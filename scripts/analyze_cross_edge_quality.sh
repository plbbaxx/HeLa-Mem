#!/usr/bin/env bash
set -euo pipefail

DATA_PATH="data/longmemeval_s.json"
MEM_DIR="artifacts/longmemeval_qwen3_4b/run_500/encoded"
BASE_PREDICTIONS="artifacts/longmemeval_qwen3_4b/run_500/predictions.jsonl"
OUTPUT="artifacts/cross_edge_quality_diagnostic/quality.json"
MODE="llm"
WORKERS=4
while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-path) DATA_PATH="$2"; shift 2 ;;
    --mem-dir) MEM_DIR="$2"; shift 2 ;;
    --base-predictions) BASE_PREDICTIONS="$2"; shift 2 ;;
    --output) OUTPUT="$2"; shift 2 ;;
    --mode) MODE="$2"; shift 2 ;;
    --workers) WORKERS="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

python -m hela_mem.analyze_cross_edge_quality \
  --data-path "$DATA_PATH" \
  --mem-dir "$MEM_DIR" \
  --base-predictions "$BASE_PREDICTIONS" \
  --output "$OUTPUT" \
  --mode "$MODE" \
  --workers "$WORKERS" \
  --top-k 15
