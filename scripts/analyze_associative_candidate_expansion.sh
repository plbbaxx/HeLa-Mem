#!/usr/bin/env bash
set -euo pipefail

DATA_PATH="data/longmemeval_s.json"
MEM_DIR="artifacts/longmemeval_qwen3_4b/run_500/encoded"
BASE_PREDICTIONS="artifacts/longmemeval_qwen3_4b/run_500/predictions.jsonl"
OUTPUT="artifacts/associative_candidate_expansion/coverage.json"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-path) DATA_PATH="$2"; shift 2 ;;
    --mem-dir) MEM_DIR="$2"; shift 2 ;;
    --base-predictions) BASE_PREDICTIONS="$2"; shift 2 ;;
    --output) OUTPUT="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

python -m hela_mem.analyze_associative_candidate_expansion \
  --data-path "$DATA_PATH" \
  --mem-dir "$MEM_DIR" \
  --base-predictions "$BASE_PREDICTIONS" \
  --output "$OUTPUT" \
  --top-k 15 \
  --seed-ks 3 5 8 \
  --neighbor-ks 2 3 5 \
  --edge-thresholds 0.1 0.2 0.3
