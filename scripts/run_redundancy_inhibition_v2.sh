#!/usr/bin/env bash
set -euo pipefail

SOURCE_ENCODED="artifacts/longmemeval_qwen3_4b/run_500/encoded"
OUTPUT_ROOT="artifacts/redundancy_inhibition_v2"
CONCURRENCY="4"
DATA_PATH="data/longmemeval_s.json"
GAMMAS=(0.1 0.2 0.3)
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

RUN_ROOTS=("$OUTPUT_ROOT/baseline")
for gamma in "${GAMMAS[@]}"; do RUN_ROOTS+=("$OUTPUT_ROOT/gamma_${gamma}"); done
for root in "${RUN_ROOTS[@]}"; do
  destination="$root/run_500/encoded"
  if [[ -d "$destination" ]]; then
    echo "Reusing existing paired snapshot for resume: $destination"
    continue
  fi
  mkdir -p "$(dirname "$destination")"
  cp -a "$SOURCE_ENCODED" "$destination"
done

COMMON_ARGS=(--stage eval --num-items 500 --concurrency "$CONCURRENCY" --data-path "$DATA_PATH")
bash scripts/run_longmemeval_qwen3_4b.sh \
  "${COMMON_ARGS[@]}" --artifact-root "$OUTPUT_ROOT/baseline"
for gamma in "${GAMMAS[@]}"; do
  bash scripts/run_longmemeval_qwen3_4b.sh \
    "${COMMON_ARGS[@]}" --artifact-root "$OUTPUT_ROOT/gamma_${gamma}" \
    --use-redundancy-inhibition --inhibition-gamma "$gamma"
done

python -m hela_mem.compare_redundancy_inhibition \
  --baseline "$OUTPUT_ROOT/baseline/run_500/predictions.jsonl" \
  --variant "gamma_0.1=$OUTPUT_ROOT/gamma_0.1/run_500/predictions.jsonl" \
  --variant "gamma_0.2=$OUTPUT_ROOT/gamma_0.2/run_500/predictions.jsonl" \
  --variant "gamma_0.3=$OUTPUT_ROOT/gamma_0.3/run_500/predictions.jsonl" \
  --output "$OUTPUT_ROOT/comparison.json"
