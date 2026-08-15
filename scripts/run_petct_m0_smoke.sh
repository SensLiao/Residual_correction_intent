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

GATE_TOOL="${SCRIPT_DIR}/validate_petct_m0_smoke.py"
RUNNER="${SCRIPT_DIR}/run_petct_m0_one_epoch.py"
PREPROCESS_GATE_TOOL="${SCRIPT_DIR}/validate_petct_m0_preprocess.py"
PREPROCESS_READY="${EXP_ROOT}/manifests/PREPROCESS_READY.json"
SMOKE_READY="${EXP_ROOT}/manifests/SMOKE_READY.json"
LEGACY_SMOKE_MARKER="${EXP_ROOT}/manifests/SMOKE_READY.done"
SMOKE_RUNS="${EXP_ROOT}/smoke_runs"
LOCK_ROOT="${EXP_ROOT}/locks"

mkdir -p "${SMOKE_RUNS}" "${LOCK_ROOT}"
if ! command -v flock >/dev/null 2>&1; then
  echo "Missing required flock executable" >&2
  exit 3
fi
exec 9>"${LOCK_ROOT}/m0_fold0_one_epoch_smoke.lock"
if ! flock -n 9; then
  echo "Another M0 smoke gate holds the lock" >&2
  exit 4
fi

for required_file in \
  "${GATE_TOOL}" \
  "${RUNNER}" \
  "${PREPROCESS_GATE_TOOL}" \
  "${PREPROCESS_READY}"; do
  if [[ ! -f "${required_file}" ]]; then
    echo "Missing smoke input: ${required_file}" >&2
    exit 5
  fi
done
if [[ ! -x "${PYTHON}" ]]; then
  echo "Pinned PET/CT Python is not runnable" >&2
  exit 6
fi
for conflict in "${SMOKE_READY}" "${LEGACY_SMOKE_MARKER}"; do
  if [[ -e "${conflict}" || -L "${conflict}" ]]; then
    echo "Refusing existing smoke publication: ${conflict}" >&2
    exit 7
  fi
done

"${PYTHON}" "${PREPROCESS_GATE_TOOL}" validate-runtime "${NNUNET_SOURCE}"
"${PYTHON}" "${GATE_TOOL}" validate-preprocess-ready "${PREPROCESS_READY}"

RUN_STAGING="$(mktemp -d "${SMOKE_RUNS}/.partial-petct_m0_fold0_smoke_$(date -u +%Y%m%dT%H%M%SZ)_XXXXXX")"
RUN_ID="$(basename "${RUN_STAGING}")"
RUN_ID="${RUN_ID#.partial-}"
RUN_FINAL="${SMOKE_RUNS}/${RUN_ID}"
RUN_RECEIPT="${RUN_STAGING}/SMOKE_BUNDLE.json"
CONSOLE_LOG="${RUN_STAGING}/console.log"

report_failed_run() {
  local status=$?
  if [[ ${status} -ne 0 ]]; then
    echo "Smoke gate failed closed; no SMOKE_READY receipt was published." >&2
    echo "Run evidence remains isolated at ${RUN_STAGING} or ${RUN_FINAL}." >&2
  fi
  trap - EXIT
  exit "${status}"
}
trap report_failed_run EXIT

"${PYTHON}" "${GATE_TOOL}" stage \
  "${PREPROCESS_READY}" "${RUN_STAGING}" "${RUN_FINAL}" "${RUN_ID}" "${GPU_ID}"

PREPROCESS_RUN="$("${PYTHON}" -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["run_dir"])' "${PREPROCESS_READY}")"
export nnUNet_raw="${PREPROCESS_RUN}/nnUNet_raw"
export nnUNet_preprocessed="${PREPROCESS_RUN}/nnUNet_preprocessed"
export nnUNet_results="${RUN_STAGING}/nnUNet_results"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export TORCHINDUCTOR_COMPILE_THREADS=1

# nnU-Net v2.8.1 enables torch.compile by default. This server exposes the
# runtime driver as libcuda.so.1 but has no unversioned system libcuda.so, so
# Triton's compile-time `-lcuda` link needs the existing CUDA stub. Keep that
# stub out of LD_LIBRARY_PATH: runtime CUDA must resolve to the NVIDIA driver.
# Its exact identity is frozen in validate_petct_m0_smoke.py and the receipt.
CUDA_DRIVER_LINK_STUB="/usr/local/cuda-11.6/targets/x86_64-linux/lib/stubs/libcuda.so"
CUDA_DRIVER_LINK_STUB_SHA256="81dcabbb572826da2e9e5edcffb7ca98a1d4728f38a3892a4999dea74716f198"
CUDA_DRIVER_LINK_STUB_BYTES="58080"
CUDA_DRIVER_LINK_DIR="$(dirname "${CUDA_DRIVER_LINK_STUB}")"
if [[ ! -f "${CUDA_DRIVER_LINK_STUB}" || -L "${CUDA_DRIVER_LINK_STUB}" ]]; then
  echo "Missing regular compile-time CUDA driver stub: ${CUDA_DRIVER_LINK_STUB}" >&2
  exit 8
fi
if [[ "$(stat -c %s "${CUDA_DRIVER_LINK_STUB}")" != "${CUDA_DRIVER_LINK_STUB_BYTES}" ]]; then
  echo "CUDA driver link stub byte count mismatch" >&2
  exit 8
fi
if [[ "$(sha256sum "${CUDA_DRIVER_LINK_STUB}" | awk '{print $1}')" != "${CUDA_DRIVER_LINK_STUB_SHA256}" ]]; then
  echo "CUDA driver link stub hash mismatch" >&2
  exit 8
fi
case ":${LD_LIBRARY_PATH:-}:" in
  *":${CUDA_DRIVER_LINK_DIR}:"*)
    echo "Refusing compile-time CUDA stub directory in LD_LIBRARY_PATH" >&2
    exit 8
    ;;
esac
export LIBRARY_PATH="${CUDA_DRIVER_LINK_DIR}${LIBRARY_PATH:+:${LIBRARY_PATH}}"
export nnUNet_compile=true

# Exact smoke identity: Dataset901 / 3d_fullres / fold 0 /
# nnUNetTrainer_1epoch. The runner uses the pinned official trainer factory and
# trainer loop but intentionally skips full-case actual validation, --npz, and
# continuation so no OOF/result artifact can be published by this gate.
"${PYTHON}" "${RUNNER}" 2>&1 | tee "${CONSOLE_LOG}"

"${PYTHON}" "${GATE_TOOL}" validate-smoke \
  --preprocess-ready "${PREPROCESS_READY}" \
  --run-id "${RUN_ID}" \
  --run-root "${RUN_STAGING}" \
  --committed-run-dir "${RUN_FINAL}" \
  --receipt "${RUN_RECEIPT}" \
  --oof-root "${EXP_ROOT}/oof_predictions" \
  --oof-root "${EXP_ROOT}/oof_probabilities" \
  --result-root "${EXP_ROOT}/evaluation"
"${PYTHON}" "${GATE_TOOL}" commit-run \
  "${RUN_STAGING}" "${RUN_FINAL}" "${RUN_RECEIPT}"
"${PYTHON}" "${GATE_TOOL}" publish-smoke-ready \
  "${RUN_FINAL}" "${RUN_FINAL}/SMOKE_BUNDLE.json" "${SMOKE_READY}" \
  --oof-root "${EXP_ROOT}/oof_predictions" \
  --oof-root "${EXP_ROOT}/oof_probabilities" \
  --result-root "${EXP_ROOT}/evaluation"

trap - EXIT
echo "Fold-0 one-epoch smoke committed: ${SMOKE_READY}"
