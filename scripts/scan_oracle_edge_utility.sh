#!/usr/bin/env bash
set -euo pipefail

DATA_PATH="data/longmemeval_s.json"
MEM_DIR="artifacts/longmemeval_qwen3_4b/run_500/encoded"
BASE_PREDICTIONS="artifacts/longmemeval_qwen3_4b/run_500/predictions.jsonl"
QUALITY_REPORT="artifacts/cross_edge_quality_diagnostic/quality.json"
OUTPUT="artifacts/oracle_edge_utility_v0/retrieval_scan.json"
EMBEDDING_MODEL="${HEBBIAN_EMBEDDING_MODEL:-all-MiniLM-L6-v2}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-path) DATA_PATH="$2"; shift 2 ;;
    --mem-dir) MEM_DIR="$2"; shift 2 ;;
    --base-predictions) BASE_PREDICTIONS="$2"; shift 2 ;;
    --quality-report) QUALITY_REPORT="$2"; shift 2 ;;
    --output) OUTPUT="$2"; shift 2 ;;
    --embedding-model) EMBEDDING_MODEL="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

python -m hela_mem.scan_oracle_edge_utility \
  --data-path "$DATA_PATH" \
  --mem-dir "$MEM_DIR" \
  --base-predictions "$BASE_PREDICTIONS" \
  --quality-report "$QUALITY_REPORT" \
  --output "$OUTPUT" \
  --embedding-model "$EMBEDDING_MODEL" \
  --top-k 15 \
  --max-flipped 3 \
  --activation-alpha 0.1 \
  --spreading-threshold 0.4 \
  --support-gain 1.0 \
  --redundant-alpha 0.2 \
  --minimum-changed 10 \
  --minimum-supporting-share 0.5
