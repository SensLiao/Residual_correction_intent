#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/petct_m0_common.sh"

if [[ $# -gt 1 ]]; then
  echo "Usage: $0 [GPU_ID]" >&2
  exit 2
fi
GPU_ID="${1:-0}"
if [[ ! "${GPU_ID}" =~ ^[0-9]+$ ]]; then
  echo "GPU_ID must be a non-negative integer" >&2
  exit 2
fi

VALIDATOR="${SCRIPT_DIR}/validate_petct_m0_oof.py"
RUNNER="${SCRIPT_DIR}/run_petct_m0_oof_fold.py"
PREPROCESS_READY="${EXP_ROOT}/manifests/PREPROCESS_READY.json"
FULL_TRAIN_READY="${EXP_ROOT}/manifests/FULL_TRAIN_READY.json"
OOF_READY="${EXP_ROOT}/manifests/OOF_READY.json"
OOF_RUNS="${EXP_ROOT}/oof_runs"
LOCK_ROOT="${EXP_ROOT}/locks"

mkdir -p "${OOF_RUNS}" "${LOCK_ROOT}"
for required in \
  "${VALIDATOR}" "${RUNNER}" "${PREPROCESS_READY}" \
  "${FULL_TRAIN_READY}"; do
  if [[ ! -f "${required}" || -L "${required}" ]]; then
    echo "Missing regular OOF prerequisite: ${required}" >&2
    exit 3
  fi
done
if [[ ! -x "${PYTHON}" ]]; then
  echo "Pinned PET/CT Python is not runnable" >&2
  exit 3
fi
if [[ -e "${OOF_READY}" || -L "${OOF_READY}" ]]; then
  echo "OOF_READY already exists; refusing overwrite" >&2
  exit 4
fi
if ! command -v flock >/dev/null 2>&1; then
  echo "Missing required flock executable" >&2
  exit 4
fi
exec 9>"${LOCK_ROOT}/m0_oof_launcher.lock"
if ! flock -n 9; then
  echo "Another M0 OOF launcher is active" >&2
  exit 5
fi

# Resolve the exact splits_final path already frozen into the aggregate
# FULL_TRAIN_READY campaign spec; the validator rehashes it against both the
# full-training and preprocessing receipts before any inference is allowed.
SPLITS_FINAL="$("${PYTHON}" -c 'import json,sys; ready=json.load(open(sys.argv[1], encoding="utf-8")); spec=json.load(open(ready["campaign_spec"]["path"], encoding="utf-8")); print(spec["prerequisite_paths"]["splits_final"])' "${FULL_TRAIN_READY}")"
if [[ ! -f "${SPLITS_FINAL}" || -L "${SPLITS_FINAL}" ]]; then
  echo "Authoritative splits_final from FULL_TRAIN_READY is unavailable" >&2
  exit 6
fi

RUN_STAGING="$(mktemp -d "${OOF_RUNS}/.partial-petct_m0_oof_$(date -u +%Y%m%dT%H%M%SZ)_XXXXXX")"
RUN_ID="$(basename "${RUN_STAGING}")"
RUN_ID="${RUN_ID#.partial-}"
RUN_FINAL="${OOF_RUNS}/${RUN_ID}"
RUN_BUNDLE="${RUN_STAGING}/OOF_BUNDLE.json"

report_failed_run() {
  local status=$?
  if [[ ${status} -ne 0 ]]; then
    echo "OOF gate failed closed; no OOF_READY was published." >&2
    echo "Run evidence remains isolated at ${RUN_STAGING} or ${RUN_FINAL}." >&2
  fi
  trap - EXIT
  exit "${status}"
}
trap report_failed_run EXIT

"${PYTHON}" "${VALIDATOR}" stage \
  "${PREPROCESS_READY}" "${SPLITS_FINAL}" \
  "${RUN_STAGING}" "${RUN_FINAL}" "${RUN_ID}" \
  "${FULL_TRAIN_READY}"

PREPROCESS_RUN="$("${PYTHON}" -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["run_dir"])' "${PREPROCESS_READY}")"
export nnUNet_raw="${PREPROCESS_RUN}/nnUNet_raw"
export nnUNet_preprocessed="${PREPROCESS_RUN}/nnUNet_preprocessed"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export nnUNet_compile=false
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# Every call is locked to its own held-out fold by the staged plan.  A fold
# receipt is no-clobber; any partial fold prevents global OOF publication.
for FOLD in 0 1 2 3 4; do
  "${PYTHON}" "${RUNNER}" "${RUN_STAGING}" "${FOLD}" \
    --device cuda:0 --source-root "${NNUNET_SOURCE}"
done

"${PYTHON}" "${VALIDATOR}" validate-oof \
  "${RUN_STAGING}" "${PREPROCESS_READY}" "${RUN_FINAL}" \
  "${RUN_ID}" "${RUN_BUNDLE}" "${FULL_TRAIN_READY}"
"${PYTHON}" "${VALIDATOR}" commit-run \
  "${RUN_STAGING}" "${RUN_FINAL}" "${RUN_BUNDLE}"
"${PYTHON}" "${VALIDATOR}" publish \
  "${RUN_FINAL}" "${RUN_FINAL}/OOF_BUNDLE.json" "${OOF_READY}"

trap - EXIT
echo "Patient-excluded 5-fold OOF committed: ${OOF_READY}"
