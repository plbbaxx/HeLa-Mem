#!/usr/bin/env bash
set -euo pipefail

MODE="raw"
SCORE_MODE="actr"
NUM_ITEMS="10"
CONCURRENCY="1"
DATA_PATH="data/longmemeval_s.json"
MEM_DIR="artifacts/longmemeval_qwen3_4b/run_500/encoded"
OUTPUT_ROOT="artifacts/raw_actr_smoke_10"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --retrieval-mode) MODE="$2"; shift 2 ;;
    --actr-score-mode) SCORE_MODE="$2"; shift 2 ;;
    --num-items) NUM_ITEMS="$2"; shift 2 ;;
    --concurrency) CONCURRENCY="$2"; shift 2 ;;
    --data-path) DATA_PATH="$2"; shift 2 ;;
    --mem-dir) MEM_DIR="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

case "$MODE" in raw|cue_idf|actr) ;; *) echo "mode must be raw, cue_idf, or actr" >&2; exit 2 ;; esac
LABEL="$MODE"
if [[ "$MODE" == "actr" ]]; then LABEL="actr_${SCORE_MODE}"; fi
RESULTS_DIR="${OUTPUT_ROOT}/eval_results_${LABEL}"
CUE_CACHE_DIR="${OUTPUT_ROOT}/cue_cache"
mkdir -p "$RESULTS_DIR" "$CUE_CACHE_DIR"

export OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://127.0.0.1:18000/v1}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
export HEBBIAN_GENERATION_MODEL="${HEBBIAN_GENERATION_MODEL:-Qwen3-4B-Instruct-2507}"
export HEBBIAN_JUDGE_MODEL="${HEBBIAN_JUDGE_MODEL:-Qwen3-4B-Instruct-2507}"
export HEBBIAN_EMBEDDING_MODEL="${HEBBIAN_EMBEDDING_MODEL:-/mnt/disk2/caoxue/models/all-MiniLM-L6-v2}"
export HEBBIAN_ENABLE_THINKING="${HEBBIAN_ENABLE_THINKING:-false}"

python -m hela_mem.eval_longmemeval \
  --data_path "$DATA_PATH" \
  --mem_dir "$MEM_DIR" \
  --results_dir "$RESULTS_DIR" \
  --num_items "$NUM_ITEMS" \
  --concurrency "$CONCURRENCY" \
  --top_k 15 \
  --semantic_top_k 5 \
  --retrieval_mode "$MODE" \
  --actr_candidate_k 30 \
  --actr_fan_threshold 0.5 \
  --actr_score_mode "$SCORE_MODE" \
  --actr_alpha 0.5 \
  --actr_cue_cache_dir "$CUE_CACHE_DIR" \
  --generation_temperature 0 \
  --judge_temperature 0
