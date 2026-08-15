#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/petct_m0_common.sh"

GATE_TOOL="${SCRIPT_DIR}/prepare_nnunet_m0_dataset.py"
AUDIT_TOOL="${SCRIPT_DIR}/audit_psma_v3_dataset.py"
EXTRACTION_MANIFEST="${EXTRACT_ROOT}/SHA256-MANIFEST.json"
MIGRATION_RECEIPT="${PETCT_ROOT}/receipts/project_migration_20260717.json"
AUDIT_POINTER="${EXP_ROOT}/audits/DATASET_AUDIT_PASS.done"
ENV_RECEIPT="${EXP_ROOT}/envs/petct_nnunet_v281.json"
AUTOPETV_RUN_DIR="${AUTOPETV_SOURCE}/nnunet-baseline/nnUNet_results/Dataset998_AutoPETV/nnUNetTrainer__nnUNetPlans__3d_fullres"
AUTOPETV_PLANS="${AUTOPETV_RUN_DIR}/plans.json"
AUTOPETV_DATASET="${AUTOPETV_RUN_DIR}/dataset.json"
PLANNING_RUNS="${EXP_ROOT}/planning_runs"
LOCK_ROOT="${EXP_ROOT}/locks"
READY_RECEIPT="${EXP_ROOT}/manifests/PLANNING_READY.json"
LEGACY_PREPROCESS_MARKER="${EXP_ROOT}/manifests/PREPROCESS_READY.done"
PREPROCESS_RECEIPT="${EXP_ROOT}/manifests/PREPROCESS_READY.json"

mkdir -p "${PLANNING_RUNS}" "${LOCK_ROOT}"
if ! command -v flock >/dev/null 2>&1; then
  echo "Missing required flock executable" >&2
  exit 2
fi
exec 9>"${LOCK_ROOT}/m0_planning.lock"
if ! flock -n 9; then
  echo "Another M0 planning gate holds the lock" >&2
  exit 3
fi

for conflict in \
  "${READY_RECEIPT}" \
  "${LEGACY_PREPROCESS_MARKER}" \
  "${PREPROCESS_RECEIPT}" \
  "${EXP_ROOT}/nnUNet_raw/${DATASET_NAME}" \
  "${EXP_ROOT}/nnUNet_preprocessed/${DATASET_NAME}"; do
  if [[ -e "${conflict}" || -L "${conflict}" ]]; then
    echo "Refusing conflicting or previously published state: ${conflict}" >&2
    exit 4
  fi
done

for required_file in \
  "${GATE_TOOL}" \
  "${AUDIT_TOOL}" \
  "${SOURCE_DATASET}/dataset.json" \
  "${SOURCE_DATASET}/splits_final.json" \
  "${SOURCE_DATASET}/psma_metadata.csv" \
  "${EXTRACTION_MANIFEST}" \
  "${MIGRATION_RECEIPT}" \
  "${AUDIT_POINTER}" \
  "${ENV_RECEIPT}" \
  "${AUTOPETV_PLANS}" \
  "${AUTOPETV_DATASET}"; do
  if [[ ! -f "${required_file}" ]]; then
    echo "Missing required planning evidence: ${required_file}" >&2
    exit 5
  fi
done
if [[ ! -x "${PYTHON}" || ! -x "${CONDA_EXE}" ]]; then
  echo "Pinned PET/CT Python or Conda executable is not runnable" >&2
  exit 6
fi

RUN_STAGING="$(mktemp -d "${PLANNING_RUNS}/.partial-psma_m0_plan_$(date -u +%Y%m%dT%H%M%SZ)_XXXXXX")"
RUN_ID="$(basename "${RUN_STAGING}")"
RUN_ID="${RUN_ID#.partial-}"
RUN_FINAL="${PLANNING_RUNS}/${RUN_ID}"

report_failed_run() {
  local status=$?
  if [[ ${status} -ne 0 ]]; then
    echo "Planning gate failed closed; no readiness receipt was published." >&2
    echo "Run evidence remains isolated at ${RUN_STAGING} or ${RUN_FINAL}." >&2
  fi
  trap - EXIT
  exit "${status}"
}
trap report_failed_run EXIT

RUN_RAW_ROOT="${RUN_STAGING}/nnUNet_raw"
RUN_PREPROCESSED_ROOT="${RUN_STAGING}/nnUNet_preprocessed"
RUN_RESULTS_ROOT="${RUN_STAGING}/nnUNet_results"
RAW_DATASET="${RUN_RAW_ROOT}/${DATASET_NAME}"
PREPROCESSED_DATASET="${RUN_PREPROCESSED_ROOT}/${DATASET_NAME}"
RUN_OWNER="${RUN_STAGING}/RUN_OWNER.json"
LIVE_CONDA_SNAPSHOT="${RUN_STAGING}/live_conda_snapshot.txt"
LIVE_PIP_SNAPSHOT="${RUN_STAGING}/live_pip_snapshot.txt"
PREFLIGHT_RECEIPT="${RUN_STAGING}/PREFLIGHT.json"
RUNTIME_IDENTITY="${RUN_STAGING}/nnunet_runtime_identity.json"
RUN_RECEIPT="${RUN_STAGING}/PLANNING_BUNDLE.json"

"${PYTHON}" "${GATE_TOOL}" write-run-owner \
  "${RUN_ID}" "${RUN_STAGING}" "${RUN_OWNER}"
"${CONDA_EXE}" list --prefix "${CONDA_ENV}" --explicit \
  > "${LIVE_CONDA_SNAPSHOT}"
"${PYTHON}" -m pip freeze --all > "${LIVE_PIP_SNAPSHOT}"

# This preflight uses file/metadata inspection only. It must pass before the
# process imports nnU-Net or executes its planning entry point.
"${PYTHON}" "${GATE_TOOL}" preflight \
  --source-dataset-root "${SOURCE_DATASET}" \
  --extraction-manifest "${EXTRACTION_MANIFEST}" \
  --migration-receipt "${MIGRATION_RECEIPT}" \
  --audit-pointer "${AUDIT_POINTER}" \
  --audits-root "${EXP_ROOT}/audits" \
  --env-receipt "${ENV_RECEIPT}" \
  --live-conda-snapshot "${LIVE_CONDA_SNAPSHOT}" \
  --live-pip-snapshot "${LIVE_PIP_SNAPSHOT}" \
  --nnunet-source "${NNUNET_SOURCE}" \
  --autopetv-plans "${AUTOPETV_PLANS}" \
  --autopetv-dataset "${AUTOPETV_DATASET}" \
  --audit-tool "${AUDIT_TOOL}" \
  --python-executable "${PYTHON}" \
  --output "${PREFLIGHT_RECEIPT}"

mkdir -p "${RAW_DATASET}" "${RUN_PREPROCESSED_ROOT}" "${RUN_RESULTS_ROOT}"
ln -s "${SOURCE_DATASET}/imagesTr" "${RAW_DATASET}/imagesTr"
ln -s "${SOURCE_DATASET}/labelsTr" "${RAW_DATASET}/labelsTr"
"${PYTHON}" "${GATE_TOOL}" write-dataset-json \
  "${SOURCE_DATASET}/dataset.json" \
  "${RAW_DATASET}/dataset.json"
"${PYTHON}" "${GATE_TOOL}" capture-runtime \
  "${NNUNET_SOURCE}" \
  --env-receipt "${ENV_RECEIPT}" \
  --python-executable "${PYTHON}" \
  --output "${RUNTIME_IDENTITY}"

export nnUNet_raw="${RUN_RAW_ROOT}"
export nnUNet_preprocessed="${RUN_PREPROCESSED_ROOT}"
export nnUNet_results="${RUN_RESULTS_ROOT}"
"${PYTHON}" -m nnunetv2.experiment_planning.plan_and_preprocess_entrypoints \
  -d "${DATASET_ID}" \
  --verify_dataset_integrity \
  -npfp 4 \
  --no_pp \
  --clean

if [[ ! -d "${PREPROCESSED_DATASET}" ]]; then
  echo "Planner did not create isolated dataset metadata" >&2
  exit 7
fi
cp --no-clobber "${SOURCE_DATASET}/splits_final.json" \
  "${PREPROCESSED_DATASET}/splits_final.json"

"${PYTHON}" "${GATE_TOOL}" validate-planning-bundle \
  --run-id "${RUN_ID}" \
  --run-root "${RUN_STAGING}" \
  --committed-run-dir "${RUN_FINAL}" \
  --source-dataset-root "${SOURCE_DATASET}" \
  --derived-dataset-root "${RAW_DATASET}" \
  --preprocessed-dataset-root "${PREPROCESSED_DATASET}" \
  --extraction-manifest "${EXTRACTION_MANIFEST}" \
  --migration-receipt "${MIGRATION_RECEIPT}" \
  --audit-pointer "${AUDIT_POINTER}" \
  --audits-root "${EXP_ROOT}/audits" \
  --env-receipt "${ENV_RECEIPT}" \
  --preflight-receipt "${PREFLIGHT_RECEIPT}" \
  --runtime-identity "${RUNTIME_IDENTITY}" \
  --nnunet-source "${NNUNET_SOURCE}" \
  --autopetv-plans "${AUTOPETV_PLANS}" \
  --autopetv-dataset "${AUTOPETV_DATASET}" \
  --audit-tool "${AUDIT_TOOL}" \
  --python-executable "${PYTHON}" \
  --run-owner "${RUN_OWNER}" \
  --receipt "${RUN_RECEIPT}"

"${PYTHON}" "${GATE_TOOL}" commit-run \
  "${RUN_STAGING}" "${RUN_FINAL}" "${RUN_RECEIPT}"
"${PYTHON}" "${GATE_TOOL}" publish-planning-ready \
  "${RUN_FINAL}" \
  "${RUN_FINAL}/PLANNING_BUNDLE.json" \
  "${READY_RECEIPT}"

trap - EXIT
echo "Planning-only gate committed: ${READY_RECEIPT}"
