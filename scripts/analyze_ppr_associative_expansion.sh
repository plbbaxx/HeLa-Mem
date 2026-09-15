#!/usr/bin/env bash
set -euo pipefail

DATA_PATH="data/longmemeval_s.json"
MEM_DIR="artifacts/longmemeval_qwen3_4b/run_500/encoded"
BASE_PREDICTIONS="artifacts/longmemeval_qwen3_4b/run_500/predictions.jsonl"
QUALITY_REPORT="artifacts/cross_edge_quality_diagnostic/quality.json"
OUTPUT="artifacts/ppr_associative_expansion_diagnostic/replay.json"
PPR_DAMPING="0.5"
PPR_TOP_N="10"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-path) DATA_PATH="$2"; shift 2 ;;
    --mem-dir) MEM_DIR="$2"; shift 2 ;;
    --base-predictions) BASE_PREDICTIONS="$2"; shift 2 ;;
    --quality-report) QUALITY_REPORT="$2"; shift 2 ;;
    --output) OUTPUT="$2"; shift 2 ;;
    --ppr-damping) PPR_DAMPING="$2"; shift 2 ;;
    --ppr-top-n) PPR_TOP_N="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

mkdir -p "$(dirname "${OUTPUT}")"
QUALITY_ARGS=()
if [[ -f "${QUALITY_REPORT}" ]]; then
  QUALITY_ARGS=(--quality-report "${QUALITY_REPORT}")
fi

python -m hela_mem.analyze_ppr_associative_expansion \
  --data-path "${DATA_PATH}" \
  --mem-dir "${MEM_DIR}" \
  --base-predictions "${BASE_PREDICTIONS}" \
  "${QUALITY_ARGS[@]}" \
  --output "${OUTPUT}" \
  --top-k 15 \
  --ppr-damping "${PPR_DAMPING}" \
  --ppr-top-n "${PPR_TOP_N}" \
  --one-hop-seed-k 8 \
  --one-hop-neighbor-k 2
