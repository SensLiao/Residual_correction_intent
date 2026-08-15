#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../common/petct_m0_common.sh"

# Two concurrent folds share a 12-logical-CPU host. Keep augmentation workers
# bounded while allowing an explicit caller override.
export nnUNet_n_proc_DA="${PETCT_NNUNET_N_PROC_DA:-4}"

if [[ $# -ne 3 ]]; then
  echo "Usage: $0 <campaign-id> <fold:0-4> <gpu-id>" >&2
  exit 2
fi
CAMPAIGN_ID="$1"
FOLD="$2"
GPU_ID="$3"
if [[ ! "${CAMPAIGN_ID}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$ ]]; then
  echo "Unsafe campaign id" >&2
  exit 2
fi
if [[ ! "${FOLD}" =~ ^[0-4]$ || ! "${GPU_ID}" =~ ^[0-9]+$ ]]; then
  echo "Fold must be 0-4 and GPU id must be a non-negative integer" >&2
  exit 2
fi

VALIDATOR="${SCRIPT_DIR}/validate_petct_m0_full_training.py"
RUNNER="${SCRIPT_DIR}/run_petct_m0_full_fold.py"
PREPROCESS_READY="${EXP_ROOT}/manifests/PREPROCESS_READY.json"
SMOKE_READY="${EXP_ROOT}/manifests/SMOKE_READY.json"
INFERENCE_SMOKE_READY="${EXP_ROOT}/manifests/INFERENCE_SMOKE_READY.json"
FULL_TRAIN_READY="${EXP_ROOT}/manifests/FULL_TRAIN_READY.json"
CAMPAIGN_ROOT="${EXP_ROOT}/full_training_runs/${CAMPAIGN_ID}"
LOCK_ROOT="${EXP_ROOT}/locks"

for required in \
  "${VALIDATOR}" "${RUNNER}" "${PREPROCESS_READY}" "${SMOKE_READY}" \
  "${INFERENCE_SMOKE_READY}"; do
  if [[ ! -f "${required}" ]]; then
    echo "Missing full-training prerequisite: ${required}" >&2
    exit 3
  fi
done
if [[ ! -x "${PYTHON}" || ! -d "${CAMPAIGN_ROOT}" ]]; then
  echo "Pinned Python or campaign root is unavailable" >&2
  exit 3
fi
if [[ -e "${FULL_TRAIN_READY}" || -L "${FULL_TRAIN_READY}" ]]; then
  echo "Full training is already published; refusing another fold run" >&2
  exit 4
fi
if ! command -v flock >/dev/null 2>&1; then
  echo "Missing required flock executable" >&2
  exit 4
fi
mkdir -p "${LOCK_ROOT}"
exec 8>"${LOCK_ROOT}/m0_gpu_${GPU_ID}.lock"
if ! flock -n 8; then
  echo "GPU ${GPU_ID} already has a full-training fold process" >&2
  exit 5
fi
exec 9>"${CAMPAIGN_ROOT}/locks/fold_${FOLD}.lock"
if ! flock -n 9; then
  echo "Fold ${FOLD} is already running" >&2
  exit 5
fi

"${PYTHON}" "${VALIDATOR}" validate-campaign "${CAMPAIGN_ROOT}" >/dev/null
ACTION_JSON="$("${PYTHON}" "${VALIDATOR}" fold-action "${CAMPAIGN_ROOT}" "${FOLD}")"
ACTION="$("${PYTHON}" -c 'import json,sys; print(json.loads(sys.argv[1])["action"])' "${ACTION_JSON}")"
if [[ "${ACTION}" == "SKIP_VERIFIED" ]]; then
  echo "Fold ${FOLD} already has a verified receipt; skipping."
  exit 0
fi
if [[ "${ACTION}" == "RESUME" ]]; then
  RESUME=true
else
  RESUME=false
fi

PREPROCESS_RUN="$("${PYTHON}" -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["run_dir"])' "${PREPROCESS_READY}")"
ACTUAL_VALIDATION="$("${PYTHON}" -c 'import json,sys; print(str(json.load(open(sys.argv[1], encoding="utf-8"))["training_contract"]["actual_validation"]).lower())' "${CAMPAIGN_ROOT}/CAMPAIGN_SPEC.json")"
EXPORT_PROBABILITIES="$("${PYTHON}" -c 'import json,sys; print(str(json.load(open(sys.argv[1], encoding="utf-8"))["training_contract"]["export_probabilities"]).lower())' "${CAMPAIGN_ROOT}/CAMPAIGN_SPEC.json")"
COMPILE_MODE="$("${PYTHON}" -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["training_contract"]["compile_contract"]["mode"])' "${CAMPAIGN_ROOT}/CAMPAIGN_SPEC.json")"
CUDA_STUB_DIR="$("${PYTHON}" -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["training_contract"]["compile_contract"]["cuda_stub_dir"])' "${CAMPAIGN_ROOT}/CAMPAIGN_SPEC.json")"
CUDA_STUB_SHA256="$("${PYTHON}" -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["training_contract"]["compile_contract"]["cuda_stub_libcuda_sha256"] or "")' "${CAMPAIGN_ROOT}/CAMPAIGN_SPEC.json")"

export nnUNet_raw="${PREPROCESS_RUN}/nnUNet_raw"
export nnUNet_preprocessed="${PREPROCESS_RUN}/nnUNet_preprocessed"
export nnUNet_results="${CAMPAIGN_ROOT}/nnUNet_results"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export TORCHINDUCTOR_COMPILE_THREADS=1

ATTEMPT_DIR="${CAMPAIGN_ROOT}/logs/fold_${FOLD}"
mkdir -p "${ATTEMPT_DIR}"
ATTEMPT_LOG="$(mktemp "${ATTEMPT_DIR}/attempt_$(date -u +%Y%m%dT%H%M%SZ)_XXXXXX.log")"
RUNTIME_RECEIPT="${ATTEMPT_LOG%.log}.runtime.json"

"${PYTHON}" "${RUNNER}" \
  --fold "${FOLD}" \
  --resume "${RESUME}" \
  --actual-validation "${ACTUAL_VALIDATION}" \
  --export-probabilities "${EXPORT_PROBABILITIES}" \
  --compile-mode "${COMPILE_MODE}" \
  --cuda-stub-dir "${CUDA_STUB_DIR}" \
  --cuda-stub-sha256 "${CUDA_STUB_SHA256}" \
  --runtime-receipt "${RUNTIME_RECEIPT}" \
  2>&1 | tee -a "${ATTEMPT_LOG}"

"${PYTHON}" "${VALIDATOR}" validate-fold "${CAMPAIGN_ROOT}" "${FOLD}" \
  --oof-root "${EXP_ROOT}/oof_predictions" \
  --oof-root "${EXP_ROOT}/oof_probabilities" \
  --result-root "${EXP_ROOT}/evaluation"
echo "Fold ${FOLD} training receipt committed."
