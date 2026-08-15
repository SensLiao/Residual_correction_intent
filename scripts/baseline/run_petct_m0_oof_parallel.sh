#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../common/petct_m0_common.sh"

if [[ $# -gt 2 ]]; then
  echo "Usage: $0 [GPU0_ID] [GPU1_ID]" >&2
  exit 2
fi
GPU0_ID="${1:-0}"
GPU1_ID="${2:-1}"
if [[ ! "${GPU0_ID}" =~ ^[0-9]+$ || ! "${GPU1_ID}" =~ ^[0-9]+$ ]]; then
  echo "GPU ids must be non-negative integers" >&2
  exit 2
fi
if [[ "${GPU0_ID}" == "${GPU1_ID}" ]]; then
  echo "Parallel OOF requires two distinct GPU ids" >&2
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

SPLITS_FINAL="$("${PYTHON}" -c 'import json,sys; ready=json.load(open(sys.argv[1], encoding="utf-8")); spec=json.load(open(ready["campaign_spec"]["path"], encoding="utf-8")); print(spec["prerequisite_paths"]["splits_final"])' "${FULL_TRAIN_READY}")"
if [[ ! -f "${SPLITS_FINAL}" || -L "${SPLITS_FINAL}" ]]; then
  echo "Authoritative splits_final from FULL_TRAIN_READY is unavailable" >&2
  exit 6
fi
OOF_HANDOFF_AVAILABLE="$("${PYTHON}" -c 'import json,sys; ready=json.load(open(sys.argv[1], encoding="utf-8")); ok=(ready.get("oof_handoff_inputs_present") is True and ready.get("actual_inference_gate_required") is False and ready.get("training_contract", {}).get("actual_validation") is True and ready.get("training_contract", {}).get("export_probabilities") is True); print("true" if ok else "false")' "${FULL_TRAIN_READY}")"

RUN_STAGING="$(mktemp -d "${OOF_RUNS}/.partial-petct_m0_oof_$(date -u +%Y%m%dT%H%M%SZ)_XXXXXX")"
RUN_ID="$(basename "${RUN_STAGING}")"
RUN_ID="${RUN_ID#.partial-}"
RUN_FINAL="${OOF_RUNS}/${RUN_ID}"
RUN_BUNDLE="${RUN_STAGING}/OOF_BUNDLE.json"

report_failed_run() {
  local status=$?
  if [[ ${status} -ne 0 ]]; then
    echo "Parallel OOF failed closed; no OOF_READY was published." >&2
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
export nnUNet_compile=false
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Dedicated inference fallback: exactly two workers are active and each worker
# executes its folds serially, so the host never launches five predictors.
run_fold_queue() {
  local gpu_id="$1"
  shift
  (
    export CUDA_VISIBLE_DEVICES="${gpu_id}"
    for fold in "$@"; do
      "${PYTHON}" "${RUNNER}" "${RUN_STAGING}" "${fold}" \
        --device cuda:0 --source-root "${NNUNET_SOURCE}"
    done
  )
}

if [[ "${OOF_HANDOFF_AVAILABLE}" == "true" ]]; then
  # The five training folds already performed their held-out validation.
  # Reuse those receipt-bound outputs on CPU; do not spend GPU time predicting
  # the same 597 cases a second time.
  for FOLD in 0 1 2 3 4; do
    "${PYTHON}" "${RUNNER}" "${RUN_STAGING}" "${FOLD}" \
      --from-actual-validation
  done
else
  GPU0_FOLDS=(0 2 4)
  GPU1_FOLDS=(1 3)
  run_fold_queue "${GPU0_ID}" "${GPU0_FOLDS[@]}" &
  GPU0_PID=$!
  run_fold_queue "${GPU1_ID}" "${GPU1_FOLDS[@]}" &
  GPU1_PID=$!

  set +e
  wait "${GPU0_PID}"
  GPU0_STATUS=$?
  wait "${GPU1_PID}"
  GPU1_STATUS=$?
  set -e
  if [[ ${GPU0_STATUS} -ne 0 || ${GPU1_STATUS} -ne 0 ]]; then
    echo "OOF fold queue failure: gpu0=${GPU0_STATUS}, gpu1=${GPU1_STATUS}" >&2
    exit 7
  fi
fi

# validate-oof reopens and rehashes all five no-clobber FOLD_DONE receipts and
# their outputs before the existing commit and atomic publication contracts.
"${PYTHON}" "${VALIDATOR}" validate-oof \
  "${RUN_STAGING}" "${PREPROCESS_READY}" "${RUN_FINAL}" \
  "${RUN_ID}" "${RUN_BUNDLE}" "${FULL_TRAIN_READY}"
"${PYTHON}" "${VALIDATOR}" commit-run \
  "${RUN_STAGING}" "${RUN_FINAL}" "${RUN_BUNDLE}"
"${PYTHON}" "${VALIDATOR}" publish \
  "${RUN_FINAL}" "${RUN_FINAL}/OOF_BUNDLE.json" "${OOF_READY}"

trap - EXIT
echo "Patient-excluded parallel 5-fold OOF committed: ${OOF_READY}"
