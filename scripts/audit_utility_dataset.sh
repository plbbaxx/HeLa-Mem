#!/usr/bin/env bash
set -euo pipefail

python -m hela_mem.audit_utility_dataset \
  --dataset-dir "${DATASET_DIR:-artifacts/utility_dataset_v1_1}" \
  --output-dir "${AUDIT_OUTPUT_DIR:-artifacts/utility_dataset_v1_1/audit}" \
  --top-extremes "${TOP_EXTREMES:-20}" \
  "$@"
