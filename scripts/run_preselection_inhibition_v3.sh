#!/usr/bin/env bash
set -euo pipefail

SOURCE_ENCODED="artifacts/longmemeval_qwen3_4b/run_500/encoded"
OUTPUT_ROOT="artifacts/preselection_inhibition_v3"
CONCURRENCY="8"
DATA_PATH="data/longmemeval_s.json"
STAGE="all"
GAMMA="0.2"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --source-encoded) SOURCE_ENCODED="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --concurrency) CONCURRENCY="$2"; shift 2 ;;
    --data-path) DATA_PATH="$2"; shift 2 ;;
    --stage) STAGE="$2"; shift 2 ;;
    --gamma) GAMMA="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
case "$STAGE" in scan|full|all) ;; *) echo "--stage must be scan, full, or all" >&2; exit 2 ;; esac

if [[ ! -d "$SOURCE_ENCODED" ]]; then
  echo "Encoded artifact directory not found: $SOURCE_ENCODED" >&2
  exit 1
fi

export HEBBIAN_TEMPERATURE="0"
export HEBBIAN_GENERATION_TEMPERATURE="0"
export HEBBIAN_EXTRACTION_TEMPERATURE="0"
export HEBBIAN_JUDGE_TEMPERATURE="0"
export HEBBIAN_MAX_FLIPPED="${HEBBIAN_MAX_FLIPPED:-3}"
export HEBBIAN_TOP_K="${HEBBIAN_TOP_K:-15}"
mkdir -p "$OUTPUT_ROOT"

if [[ "$STAGE" == "scan" || "$STAGE" == "all" ]]; then
  python -m hela_mem.scan_preselection_candidates \
    --data-path "$DATA_PATH" \
    --mem-dir "$SOURCE_ENCODED" \
    --output "$OUTPUT_ROOT/retrieval_scan.json" \
    --top-k "$HEBBIAN_TOP_K" \
    --max-flipped "$HEBBIAN_MAX_FLIPPED" \
    --concurrency "$CONCURRENCY" \
    --gammas 0.1 0.2 0.3
fi

if [[ "$STAGE" == "full" || "$STAGE" == "all" ]]; then
  RUN_ROOTS=(
    "$OUTPUT_ROOT/baseline_a"
    "$OUTPUT_ROOT/baseline_b"
    "$OUTPUT_ROOT/preselection_control"
    "$OUTPUT_ROOT/v3_gamma_${GAMMA}"
  )
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
    "${COMMON_ARGS[@]}" --artifact-root "$OUTPUT_ROOT/baseline_a"
  bash scripts/run_longmemeval_qwen3_4b.sh \
    "${COMMON_ARGS[@]}" --artifact-root "$OUTPUT_ROOT/baseline_b"
  python -m hela_mem.verify_baseline_repeat \
    --first "$OUTPUT_ROOT/baseline_a/run_500/predictions.jsonl" \
    --second "$OUTPUT_ROOT/baseline_b/run_500/predictions.jsonl" \
    --output "$OUTPUT_ROOT/deterministic_baseline_check.json"
  bash scripts/run_longmemeval_qwen3_4b.sh \
    "${COMMON_ARGS[@]}" --artifact-root "$OUTPUT_ROOT/preselection_control" \
    --use-preselection-pool
  bash scripts/run_longmemeval_qwen3_4b.sh \
    "${COMMON_ARGS[@]}" --artifact-root "$OUTPUT_ROOT/v3_gamma_${GAMMA}" \
    --use-preselection-pool --use-redundancy-inhibition --inhibition-gamma "$GAMMA"

  python -m hela_mem.compare_preselection_v3 \
    --baseline-a "$OUTPUT_ROOT/baseline_a/run_500/predictions.jsonl" \
    --baseline-b "$OUTPUT_ROOT/baseline_b/run_500/predictions.jsonl" \
    --control "$OUTPUT_ROOT/preselection_control/run_500/predictions.jsonl" \
    --v3 "$OUTPUT_ROOT/v3_gamma_${GAMMA}/run_500/predictions.jsonl" \
    --output "$OUTPUT_ROOT/comparison.json"
fi
