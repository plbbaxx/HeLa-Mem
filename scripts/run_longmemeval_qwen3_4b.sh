#!/usr/bin/env bash
set -euo pipefail

STAGE="all"
NUM_ITEMS="5"
CONCURRENCY="4"
ARTIFACT_ROOT="artifacts/longmemeval_qwen3_4b"
DATA_PATH="data/longmemeval_s.json"
USE_REDUNDANCY_INHIBITION="false"
INHIBITION_GAMMA="0.2"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --stage) STAGE="$2"; shift 2 ;;
    --num-items) NUM_ITEMS="$2"; shift 2 ;;
    --concurrency) CONCURRENCY="$2"; shift 2 ;;
    --artifact-root) ARTIFACT_ROOT="$2"; shift 2 ;;
    --data-path) DATA_PATH="$2"; shift 2 ;;
    --use-redundancy-inhibition) USE_REDUNDANCY_INHIBITION="true"; shift ;;
    --inhibition-gamma) INHIBITION_GAMMA="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

case "$STAGE" in encode|eval|all) ;; *) echo "--stage must be encode, eval, or all" >&2; exit 2 ;; esac
case "$NUM_ITEMS" in 1|5|20|100|500) ;; *) echo "--num-items must be 1, 5, 20, 100, or 500" >&2; exit 2 ;; esac

RUN_DIR="${ARTIFACT_ROOT}/run_${NUM_ITEMS}"
ENCODED_DIR="${RUN_DIR}/encoded"
EVAL_DIR="${RUN_DIR}/eval_results"
mkdir -p "$ENCODED_DIR" "$EVAL_DIR"

export OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://127.0.0.1:8000/v1}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
export HEBBIAN_GENERATION_MODEL="${HEBBIAN_GENERATION_MODEL:-Qwen3-4B-Instruct-2507}"
export HEBBIAN_EXTRACTION_MODEL="${HEBBIAN_EXTRACTION_MODEL:-Qwen3-4B-Instruct-2507}"
export HEBBIAN_JUDGE_MODEL="${HEBBIAN_JUDGE_MODEL:-Qwen3-4B-Instruct-2507}"
export HEBBIAN_EMBEDDING_MODEL="${HEBBIAN_EMBEDDING_MODEL:-all-MiniLM-L6-v2}"
export HEBBIAN_ENABLE_THINKING="${HEBBIAN_ENABLE_THINKING:-false}"
export HEBBIAN_TAU="${HEBBIAN_TAU:-1e7}"
export HEBBIAN_LEARNING_RATE="${HEBBIAN_LEARNING_RATE:-0.02}"
export HEBBIAN_DECAY_RATE="${HEBBIAN_DECAY_RATE:-0.995}"
export HEBBIAN_ACTIVATION_ALPHA="${HEBBIAN_ACTIVATION_ALPHA:-0.1}"
export HEBBIAN_SPREADING_THRESHOLD="${HEBBIAN_SPREADING_THRESHOLD:-0.4}"
export HEBBIAN_MAX_FLIPPED="${HEBBIAN_MAX_FLIPPED:-3}"
export HEBBIAN_KEYWORD_WEIGHT="${HEBBIAN_KEYWORD_WEIGHT:-0.7}"
export HEBBIAN_TOP_K="${HEBBIAN_TOP_K:-15}"
export HEBBIAN_SEMANTIC_TOP_K="${HEBBIAN_SEMANTIC_TOP_K:-5}"
export HEBBIAN_KNOWLEDGE_BUFFER_SIZE="${HEBBIAN_KNOWLEDGE_BUFFER_SIZE:-10}"
export HEBBIAN_USE_REDUNDANCY_INHIBITION="$USE_REDUNDANCY_INHIBITION"
export HEBBIAN_INHIBITION_GAMMA="$INHIBITION_GAMMA"

if [[ "$STAGE" == "encode" || "$STAGE" == "all" ]]; then
  python -m hela_mem.encode_longmemeval \
    --data_path "$DATA_PATH" --output_dir "$ENCODED_DIR" \
    --num_items "$NUM_ITEMS" --concurrency "$CONCURRENCY"
fi

if [[ "$STAGE" == "eval" || "$STAGE" == "all" ]]; then
  CONSOLIDATION_ARGS=()
  if [[ "${HEBBIAN_USE_CONSOLIDATION:-false}" == "true" ]]; then CONSOLIDATION_ARGS+=(--use_consolidation); fi
  INHIBITION_ARGS=()
  if [[ "$USE_REDUNDANCY_INHIBITION" == "true" ]]; then INHIBITION_ARGS+=(--use-redundancy-inhibition); fi
  python -m hela_mem.eval_longmemeval \
    --data_path "$DATA_PATH" --mem_dir "$ENCODED_DIR" --results_dir "$EVAL_DIR" \
    --num_items "$NUM_ITEMS" --concurrency "$CONCURRENCY" \
    --top_k "$HEBBIAN_TOP_K" --semantic_top_k "$HEBBIAN_SEMANTIC_TOP_K" \
    --inhibition-gamma "$INHIBITION_GAMMA" "${INHIBITION_ARGS[@]}" \
    "${CONSOLIDATION_ARGS[@]}"
fi
