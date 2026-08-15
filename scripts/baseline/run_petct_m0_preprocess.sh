#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../common/petct_m0_common.sh"

GATE_TOOL="${SCRIPT_DIR}/validate_petct_m0_preprocess.py"
PLANNING_READY="${EXP_ROOT}/manifests/PLANNING_READY.json"
PREPROCESS_READY="${EXP_ROOT}/manifests/PREPROCESS_READY.json"
LEGACY_PREPROCESS_MARKER="${EXP_ROOT}/manifests/PREPROCESS_READY.done"
PREPROCESS_RUNS="${EXP_ROOT}/preprocess_runs"
LOCK_ROOT="${EXP_ROOT}/locks"

mkdir -p "${PREPROCESS_RUNS}" "${LOCK_ROOT}"
if ! command -v flock >/dev/null 2>&1; then
  echo "Missing required flock executable" >&2
  exit 2
fi
exec 9>"${LOCK_ROOT}/m0_preprocessing.lock"
if ! flock -n 9; then
  echo "Another M0 preprocessing gate holds the lock" >&2
  exit 3
fi

for required_file in "${GATE_TOOL}" "${PLANNING_READY}"; do
  if [[ ! -f "${required_file}" ]]; then
    echo "Missing preprocessing input: ${required_file}" >&2
    exit 4
  fi
done
if [[ ! -x "${PYTHON}" ]]; then
  echo "Pinned PET/CT Python is not runnable" >&2
  exit 5
fi
for conflict in "${PREPROCESS_READY}" "${LEGACY_PREPROCESS_MARKER}"; do
  if [[ -e "${conflict}" || -L "${conflict}" ]]; then
    echo "Refusing existing preprocessing publication: ${conflict}" >&2
    exit 6
  fi
done

"${PYTHON}" "${GATE_TOOL}" validate-planning-ready "${PLANNING_READY}"
"${PYTHON}" "${GATE_TOOL}" validate-runtime "${NNUNET_SOURCE}"

RUN_STAGING="$(mktemp -d "${PREPROCESS_RUNS}/.partial-psma_m0_preprocess_$(date -u +%Y%m%dT%H%M%SZ)_XXXXXX")"
RUN_ID="$(basename "${RUN_STAGING}")"
RUN_ID="${RUN_ID#.partial-}"
RUN_FINAL="${PREPROCESS_RUNS}/${RUN_ID}"
RUN_RECEIPT="${RUN_STAGING}/PREPROCESSING_BUNDLE.json"

report_failed_run() {
  local status=$?
  if [[ ${status} -ne 0 ]]; then
    echo "Preprocessing gate failed closed; no PREPROCESS_READY receipt was published." >&2
    echo "Run evidence remains isolated at ${RUN_STAGING} or ${RUN_FINAL}." >&2
  fi
  trap - EXIT
  exit "${status}"
}
trap report_failed_run EXIT

"${PYTHON}" "${GATE_TOOL}" stage \
  "${PLANNING_READY}" "${RUN_STAGING}" "${RUN_FINAL}" "${RUN_ID}"

export nnUNet_raw="${RUN_STAGING}/nnUNet_raw"
export nnUNet_preprocessed="${RUN_STAGING}/nnUNet_preprocessed"
export nnUNet_results="${RUN_STAGING}/nnUNet_results"

# Call the pinned nnU-Net v2.8.1 API directly. This performs preprocessing
# only for Dataset901 / nnUNetPlans / 3d_fullres with four CPU processes.
"${PYTHON}" -c "from nnunetv2.experiment_planning.plan_and_preprocess_api import preprocess; preprocess([901], 'nnUNetPlans', ['3d_fullres'], [4], verbose=False, show_progress_bar=True)"

"${PYTHON}" "${GATE_TOOL}" validate-preprocessing \
  --planning-ready "${PLANNING_READY}" \
  --run-id "${RUN_ID}" \
  --run-root "${RUN_STAGING}" \
  --committed-run-dir "${RUN_FINAL}" \
  --receipt "${RUN_RECEIPT}"
"${PYTHON}" "${GATE_TOOL}" commit-run \
  "${RUN_STAGING}" "${RUN_FINAL}" "${RUN_RECEIPT}"
"${PYTHON}" "${GATE_TOOL}" publish-preprocess-ready \
  "${RUN_FINAL}" "${RUN_FINAL}/PREPROCESSING_BUNDLE.json" "${PREPROCESS_READY}"

trap - EXIT
echo "Preprocessing-only gate committed: ${PREPROCESS_READY}"
