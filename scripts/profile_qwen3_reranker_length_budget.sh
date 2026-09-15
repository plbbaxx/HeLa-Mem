#!/usr/bin/env bash
set -uo pipefail

ROOT_OUTPUT="${QWEN3_RERANKER_LENGTH_PROFILE_ROOT:-artifacts/qwen3_reranker_0_6b_length_budget_profiles}"
MODEL_PATH="${QWEN3_RERANKER_MODEL_PATH:-/mnt/disk2/caoxue/models/Qwen3-Reranker-0.6B}"
DATASET_DIR="${UTILITY_DATASET_DIR:-artifacts/utility_dataset_v1_1}"
ATTENTION="${QWEN3_RERANKER_ATTENTION:-sdpa}"

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
mkdir -p "${ROOT_OUTPUT}"

# Start from the proposed practical budget, then test the upper and lower bounds.
for length in 4096 8192 2048; do
  output="${ROOT_OUTPUT}/length_${length}"
  mkdir -p "${output}"
  echo "===== profiling max_length=${length} ====="
  QWEN3_RERANKER_OUTPUT_DIR="${output}" \
  QWEN3_RERANKER_MODEL_PATH="${MODEL_PATH}" \
  UTILITY_DATASET_DIR="${DATASET_DIR}" \
  QWEN3_RERANKER_MAX_LENGTH="${length}" \
  QWEN3_RERANKER_TRAIN_BATCH_SIZE=1 \
  QWEN3_RERANKER_EVAL_BATCH_SIZE=1 \
  QWEN3_RERANKER_ATTENTION="${ATTENTION}" \
    bash scripts/train_qwen3_reranker_0_6b_utility.sh --stage profile \
      2>&1 | tee "${output}/run.log"
  echo "profile_exit_code=${PIPESTATUS[0]} max_length=${length}"
done

python - "${ROOT_OUTPUT}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
rows = []
for length in (2048, 4096, 8192):
    path = root / f"length_{length}" / "profile_smoke_test.json"
    if not path.exists():
        rows.append({"max_length": length, "status": "missing_report"})
        continue
    report = json.loads(path.read_text(encoding="utf-8"))
    selected = report.get("selected_training_configuration")
    if selected is None:
        rows.append({"max_length": length, "status": "oom"})
        continue
    total = report["base_model_memory"]["device_total_bytes"] / (1024 ** 3)
    peak = float(selected["peak_gpu_memory_gib"])
    rows.append({
        "max_length": length,
        "status": "fit",
        "pair_forward_mode": selected["pair_forward_mode"],
        "gradient_checkpointing": selected["gradient_checkpointing"],
        "peak_gpu_memory_gib": peak,
        "device_total_gib": total,
        "peak_memory_fraction": peak / total,
        "safe_below_90_percent": peak / total <= 0.90,
        "seconds_per_step": selected["seconds_per_step"],
        "actual_profile_shape": selected["forward_shape_diagnostic"]["input_ids_shape"],
    })

safe = [row for row in rows if row.get("safe_below_90_percent")]
fit = [row for row in rows if row.get("status") == "fit"]
recommended = max(safe, key=lambda row: row["max_length"], default=None)
reason = "largest_fit_with_at_least_10_percent_gpu_headroom"
if recommended is None and fit:
    recommended = min(fit, key=lambda row: row["peak_memory_fraction"])
    reason = "no_profile_has_10_percent_headroom_choose_lowest_memory_fit_for_diagnosis_only"
summary = {
    "protocol": "qwen3_reranker_0_6b_length_budget_profile_v1",
    "full_training_started": False,
    "lengths_tested": [2048, 4096, 8192],
    "profiles": rows,
    "safety_rule": "peak_gpu_memory <= 90 percent of device capacity",
    "recommended_max_length": recommended["max_length"] if recommended else None,
    "recommendation_reason": reason if recommended else "no_length_fit",
}
(root / "length_budget_summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps(summary, ensure_ascii=False, indent=2))
PY
