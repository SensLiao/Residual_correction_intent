#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/petct_m0_common.sh"

test -f "${EXP_ROOT}/manifests/EXTRACTION_READY.done"
RUN_ID="psma_v3_nifti_audit_$(date +%Y%m%dT%H%M%S)"
RUN_DIR="${EXP_ROOT}/audits/${RUN_ID}"
test ! -e "${RUN_DIR}"
mkdir -p "${RUN_DIR}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
"${PYTHON}" "${SCRIPT_DIR}/audit_psma_v3_dataset.py" \
  "${SOURCE_DATASET}" \
  --output-dir "${RUN_DIR}" \
  --workers 2 \
  --expected-cases 597 \
  --expected-patients 378 \
  --expected-empty 58 \
  --expected-folds 5

test -f "${RUN_DIR}/AUDIT_COMPLETE.json"
"${PYTHON}" - "${RUN_DIR}/AUDIT_COMPLETE.json" <<'PY'
import json
import sys
from pathlib import Path

receipt = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if receipt.get("status") != "COMMITTED" or receipt.get("audit_status") != "PASS":
    raise SystemExit(f"audit receipt is not a committed PASS: {receipt}")
PY
printf 'run_dir=%s\ncompleted_at=%s\n' "${RUN_DIR}" "$(date --iso-8601=seconds)" \
  > "${EXP_ROOT}/audits/DATASET_AUDIT_PASS.done"
