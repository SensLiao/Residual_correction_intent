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

GATE_TOOL="${SCRIPT_DIR}/validate_petct_m0_inference_smoke.py"
RUNNER="${SCRIPT_DIR}/run_petct_m0_inference_smoke.py"
PREPROCESS_READY="${EXP_ROOT}/manifests/PREPROCESS_READY.json"
SMOKE_READY="${EXP_ROOT}/manifests/SMOKE_READY.json"
INFERENCE_SMOKE_READY="${EXP_ROOT}/manifests/INFERENCE_SMOKE_READY.json"
INFERENCE_RUNS="${EXP_ROOT}/inference_smoke_runs"
LOCK_ROOT="${EXP_ROOT}/locks"

mkdir -p "${INFERENCE_RUNS}" "${LOCK_ROOT}"
if ! command -v flock >/dev/null 2>&1; then
  echo "Missing required flock executable" >&2
  exit 3
fi
exec 9>"${LOCK_ROOT}/m0_fold0_inference_smoke.lock"
if ! flock -n 9; then
  echo "Another inference-smoke gate holds the lock" >&2
  exit 4
fi

for required_file in \
  "${GATE_TOOL}" \
  "${RUNNER}" \
  "${PREPROCESS_READY}" \
  "${SMOKE_READY}"; do
  if [[ ! -f "${required_file}" || -L "${required_file}" ]]; then
    echo "Missing regular inference-smoke input: ${required_file}" >&2
    exit 5
  fi
done
if [[ ! -x "${PYTHON}" ]]; then
  echo "Pinned PET/CT Python is not runnable" >&2
  exit 6
fi
if [[ -e "${INFERENCE_SMOKE_READY}" || -L "${INFERENCE_SMOKE_READY}" ]]; then
  echo "Refusing existing inference-smoke publication" >&2
  exit 7
fi

"${PYTHON}" "${GATE_TOOL}" validate-prerequisites \
  "${PREPROCESS_READY}" "${SMOKE_READY}" "${NNUNET_SOURCE}"

RUN_STAGING="$(mktemp -d "${INFERENCE_RUNS}/.partial-petct_m0_inference_smoke_$(date -u +%Y%m%dT%H%M%SZ)_XXXXXX")"
RUN_ID="$(basename "${RUN_STAGING}")"
RUN_ID="${RUN_ID#.partial-}"
RUN_FINAL="${INFERENCE_RUNS}/${RUN_ID}"
RUN_RECEIPT="${RUN_STAGING}/INFERENCE_SMOKE_BUNDLE.json"
SPEC="${RUN_STAGING}/INFERENCE_SMOKE_SPEC.json"
CONSOLE_LOG="${RUN_STAGING}/console.log"

report_failed_run() {
  local status=$?
  if [[ ${status} -ne 0 ]]; then
    echo "Inference-smoke gate failed closed; no ready receipt was published." >&2
    echo "Run evidence remains isolated at ${RUN_STAGING} or ${RUN_FINAL}." >&2
  fi
  trap - EXIT
  exit "${status}"
}
trap report_failed_run EXIT

"${PYTHON}" "${GATE_TOOL}" stage \
  "${PREPROCESS_READY}" "${SMOKE_READY}" "${NNUNET_SOURCE}" \
  "${RUN_STAGING}" "${RUN_FINAL}" "${RUN_ID}" "${GPU_ID}"

CASE_ID="$("${PYTHON}" -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["selected_case"]["case_id"])' "${SPEC}")"
MODEL_DIR="$("${PYTHON}" -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["paths"]["model_training_output_dir"])' "${SPEC}")"
SOURCE_CT="$("${PYTHON}" -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["paths"]["source_ct"])' "${SPEC}")"
SOURCE_PET="$("${PYTHON}" -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["paths"]["source_pet"])' "${SPEC}")"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export TORCHINDUCTOR_COMPILE_THREADS=1
export nnUNet_compile=false

"${PYTHON}" "${RUNNER}" \
  --model-dir "${MODEL_DIR}" \
  --case-id "${CASE_ID}" \
  --ct "${SOURCE_CT}" \
  --pet "${SOURCE_PET}" \
  --output-dir "${RUN_STAGING}/predictions" 2>&1 | tee "${CONSOLE_LOG}"

"${PYTHON}" "${GATE_TOOL}" validate-output \
  "${PREPROCESS_READY}" "${SMOKE_READY}" "${NNUNET_SOURCE}" \
  "${RUN_STAGING}" "${RUN_FINAL}" "${RUN_ID}" "${RUN_RECEIPT}" \
  --oof-root "${EXP_ROOT}/oof_predictions" \
  --oof-root "${EXP_ROOT}/oof_probabilities" \
  --result-root "${EXP_ROOT}/evaluation"
"${PYTHON}" "${GATE_TOOL}" commit-run \
  "${RUN_STAGING}" "${RUN_FINAL}" "${RUN_RECEIPT}"
"${PYTHON}" "${GATE_TOOL}" publish-ready \
  "${RUN_FINAL}" "${RUN_FINAL}/INFERENCE_SMOKE_BUNDLE.json" \
  "${INFERENCE_SMOKE_READY}" \
  --oof-root "${EXP_ROOT}/oof_predictions" \
  --oof-root "${EXP_ROOT}/oof_probabilities" \
  --result-root "${EXP_ROOT}/evaluation"

trap - EXIT
echo "Fold-0 actual-case inference smoke committed: ${INFERENCE_SMOKE_READY}"
