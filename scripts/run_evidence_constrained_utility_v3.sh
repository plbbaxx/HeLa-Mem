#!/usr/bin/env bash
set -euo pipefail

DATA_PATH="data/longmemeval_s.json"
MEM_DIR="artifacts/longmemeval_qwen3_4b/run_500/encoded"
ORACLE_V05="artifacts/exact_oracle_edge_utility_v05/retrieval_scan.json"
V2_REPORT="artifacts/structured_residual_utility_v2/structured_residual_utility_v2_report.json"
V2_GAP_DIR="artifacts/structured_residual_utility_v2/gap_items"
OUTPUT_DIR="artifacts/evidence_constrained_utility_v3"
WORKERS=8
while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-path) DATA_PATH="$2"; shift 2 ;;
    --mem-dir) MEM_DIR="$2"; shift 2 ;;
    --oracle-v05) ORACLE_V05="$2"; shift 2 ;;
    --v2-report) V2_REPORT="$2"; shift 2 ;;
    --v2-gap-dir) V2_GAP_DIR="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --workers) WORKERS="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

python -m hela_mem.evidence_constrained_utility_v3 \
  --data-path "$DATA_PATH" \
  --mem-dir "$MEM_DIR" \
  --oracle-v05 "$ORACLE_V05" \
  --v2-report "$V2_REPORT" \
  --v2-gap-dir "$V2_GAP_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --workers "$WORKERS" \
  --top-k 15
