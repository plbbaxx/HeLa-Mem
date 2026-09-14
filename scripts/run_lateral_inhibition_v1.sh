#!/usr/bin/env bash
set -euo pipefail

SOURCE_ENCODED="artifacts/longmemeval_qwen3_4b/run_500/encoded"
OUTPUT_ROOT="artifacts/lateral_inhibition_v1"
CONCURRENCY="4"
DATA_PATH="data/longmemeval_s.json"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --source-encoded) SOURCE_ENCODED="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --concurrency) CONCURRENCY="$2"; shift 2 ;;
    --data-path) DATA_PATH="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ ! -d "$SOURCE_ENCODED" ]]; then
  echo "Encoded artifact directory not found: $SOURCE_ENCODED" >&2
  exit 1
fi

BASELINE_ROOT="$OUTPUT_ROOT/baseline"
INHIBITION_ROOT="$OUTPUT_ROOT/inhibition_beta_0.15_m_7"
for destination in "$BASELINE_ROOT/run_500/encoded" "$INHIBITION_ROOT/run_500/encoded"; do
  if [[ -e "$destination" ]]; then
    echo "Refusing to overwrite existing paired snapshot: $destination" >&2
    exit 1
  fi
  mkdir -p "$(dirname "$destination")"
  cp -a "$SOURCE_ENCODED" "$destination"
done

COMMON_ARGS=(
  --stage eval --num-items 500 --concurrency "$CONCURRENCY" --data-path "$DATA_PATH"
)
bash scripts/run_longmemeval_qwen3_4b.sh \
  "${COMMON_ARGS[@]}" --artifact-root "$BASELINE_ROOT"
bash scripts/run_longmemeval_qwen3_4b.sh \
  "${COMMON_ARGS[@]}" --artifact-root "$INHIBITION_ROOT" \
  --use-inhibition --inhibition-beta 0.15 --inhibition-top-m 7

python -m hela_mem.compare_lateral_inhibition \
  --baseline "$BASELINE_ROOT/run_500/predictions.jsonl" \
  --inhibition "$INHIBITION_ROOT/run_500/predictions.jsonl" \
  --output "$OUTPUT_ROOT/comparison.json"
