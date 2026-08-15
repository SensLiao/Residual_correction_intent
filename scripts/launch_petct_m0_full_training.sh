#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/petct_m0_common.sh"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <campaign-id> [--resume] [--gpu0 ID] [--gpu1 ID] [--actual-validation true|false] [--export-probabilities true|false] [--compile-mode triton-stub-link|disabled] [--cuda-stub-dir PATH]" >&2
  exit 2
fi
CAMPAIGN_ID="$1"
shift
RESUME_CAMPAIGN=false
GPU0_ID=0
GPU1_ID=1
ACTUAL_VALIDATION=true
EXPORT_PROBABILITIES=true
COMPILE_MODE=triton-stub-link
CUDA_STUB_DIR=/usr/local/cuda-11.6/targets/x86_64-linux/lib/stubs
while [[ $# -gt 0 ]]; do
  case "$1" in
    --resume) RESUME_CAMPAIGN=true; shift ;;
    --gpu0) GPU0_ID="$2"; shift 2 ;;
    --gpu1) GPU1_ID="$2"; shift 2 ;;
    --actual-validation) ACTUAL_VALIDATION="$2"; shift 2 ;;
    --export-probabilities) EXPORT_PROBABILITIES="$2"; shift 2 ;;
    --compile-mode) COMPILE_MODE="$2"; shift 2 ;;
    --cuda-stub-dir) CUDA_STUB_DIR="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
if [[ ! "${CAMPAIGN_ID}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$ ]]; then
  echo "Unsafe campaign id" >&2
  exit 2
fi
if [[ ! "${GPU0_ID}" =~ ^[0-9]+$ || ! "${GPU1_ID}" =~ ^[0-9]+$ || "${GPU0_ID}" == "${GPU1_ID}" ]]; then
  echo "Two distinct non-negative GPU ids are required" >&2
  exit 2
fi
if [[ ! "${ACTUAL_VALIDATION}" =~ ^(true|false)$ || ! "${EXPORT_PROBABILITIES}" =~ ^(true|false)$ ]]; then
  echo "Validation switches must be true or false" >&2
  exit 2
fi
if [[ "${EXPORT_PROBABILITIES}" == "true" && "${ACTUAL_VALIDATION}" != "true" ]]; then
  echo "Probability export requires actual validation" >&2
  exit 2
fi
if [[ ! "${COMPILE_MODE}" =~ ^(triton-stub-link|disabled)$ ]]; then
  echo "Compile mode must be triton-stub-link or disabled" >&2
  exit 2
fi

VALIDATOR="${SCRIPT_DIR}/validate_petct_m0_full_training.py"
FOLD_RUNNER="${SCRIPT_DIR}/run_petct_m0_fold.sh"
PREPROCESS_READY="${EXP_ROOT}/manifests/PREPROCESS_READY.json"
SMOKE_READY="${EXP_ROOT}/manifests/SMOKE_READY.json"
INFERENCE_SMOKE_READY="${EXP_ROOT}/manifests/INFERENCE_SMOKE_READY.json"
FULL_TRAIN_READY="${EXP_ROOT}/manifests/FULL_TRAIN_READY.json"
CAMPAIGN_ROOT="${EXP_ROOT}/full_training_runs/${CAMPAIGN_ID}"
mkdir -p "${EXP_ROOT}/full_training_runs" "${EXP_ROOT}/locks"

exec 9>"${EXP_ROOT}/locks/m0_full_training_launcher.lock"
if ! flock -n 9; then
  echo "Another full-training launcher is active" >&2
  exit 3
fi
if [[ -e "${FULL_TRAIN_READY}" || -L "${FULL_TRAIN_READY}" ]]; then
  echo "FULL_TRAIN_READY already exists; refusing overwrite" >&2
  exit 4
fi

if [[ "${RESUME_CAMPAIGN}" == "true" ]]; then
  "${PYTHON}" "${VALIDATOR}" validate-campaign "${CAMPAIGN_ROOT}" >/dev/null
else
  if ! mkdir "${CAMPAIGN_ROOT}"; then
    echo "Campaign already exists; use --resume after verifying its identity" >&2
    exit 5
  fi
  "${PYTHON}" "${VALIDATOR}" init-campaign \
    --preprocess-ready "${PREPROCESS_READY}" \
    --smoke-ready "${SMOKE_READY}" \
    --inference-smoke-ready "${INFERENCE_SMOKE_READY}" \
    --source-root "${NNUNET_SOURCE}" \
    --campaign-root "${CAMPAIGN_ROOT}" \
    --campaign-id "${CAMPAIGN_ID}" \
    --actual-validation "${ACTUAL_VALIDATION}" \
    --export-probabilities "${EXPORT_PROBABILITIES}" \
    --compile-mode "${COMPILE_MODE}" \
    --cuda-stub-dir "${CUDA_STUB_DIR}"
fi

run_gpu_sequence() {
  local gpu_id="$1"
  shift
  local fold
  for fold in "$@"; do
    "${FOLD_RUNNER}" "${CAMPAIGN_ID}" "${fold}" "${gpu_id}"
  done
}

# Fold 0 is the standard-training stability gate. Only after it has a verified
# receipt do the remaining folds fan out across two serial-per-GPU workers.
"${FOLD_RUNNER}" "${CAMPAIGN_ID}" 0 "${GPU0_ID}"

run_parallel_workers() {
  run_gpu_sequence "${GPU0_ID}" 2 4 &
  local worker0=$!
  run_gpu_sequence "${GPU1_ID}" 1 3 &
  local worker1=$!
  local rc0 rc1
  set +e
  wait "${worker0}"
  rc0=$?
  wait "${worker1}"
  rc1=$?
  set -e
  if [[ ${rc0} -ne 0 || ${rc1} -ne 0 ]]; then
    echo "At least one GPU worker failed; FULL_TRAIN_READY will not be published" >&2
    return 1
  fi
}
run_parallel_workers

"${PYTHON}" "${VALIDATOR}" publish-full-ready \
  "${CAMPAIGN_ROOT}" "${FULL_TRAIN_READY}" \
  --oof-root "${EXP_ROOT}/oof_predictions" \
  --oof-root "${EXP_ROOT}/oof_probabilities" \
  --result-root "${EXP_ROOT}/evaluation"
echo "Five-fold standard training committed: ${FULL_TRAIN_READY}"
