#!/usr/bin/env bash
set -euo pipefail

# Run an explicitly selected, execution-admitted subset of the external
# spatial comparators after the natural OOF data receipt exists. A method that
# is not selected is neither validated nor touched. Each selected method keeps
# separate positive-only diagnostic and native-diagnostic fairness tables.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

usage() {
  cat >&2 <<'EOF'
Usage:
  run_petct_external_comparators_after_data.sh \
    --route-a-run-root DIR --run-root DIR --methods ID[,ID] [options]

Options:
  --methods IDS              Required explicit subset: nninteractive and/or
                             scribbleprompt (comma-separated, no default).
  --partition val|test       Must match the Route-A data run (default: val).
  --test-access-receipt FILE Required only for test; must be the consumed
                             receipt bound to --route-a-run-root.
  --gpu-scribbleprompt ID    Physical GPU for the 2D queue (default: 0).
  --gpu-nninteractive ID     Physical GPU for the 3D queue (default: 1).

Required environment:
  PETCT_AUTOPETV_METRICS     Pinned official AutoPET-V metrics.py.

Optional environment overrides:
  PETCT_CORE_PYTHON
  PETCT_SCRIBBLEPROMPT_PYTHON
  PETCT_NNINTERACTIVE_PYTHON
  PETCT_SCRIBBLEPROMPT_SOURCE
  PETCT_SCRIBBLEPROMPT_CHECKPOINT
  PETCT_NNINTERACTIVE_SOURCE
  PETCT_NNINTERACTIVE_MODEL_FOLDER
  PETCT_EXTERNAL_COMPARATOR_CONFIG
  PETCT_EXPERIMENT_CONFIG
EOF
}

ROUTE_A_RUN_ROOT=""
RUN_ROOT=""
METHODS_RAW=""
PARTITION="val"
TEST_ACCESS_RECEIPT=""
GPU_SP=0
GPU_NNI=1
while [[ $# -gt 0 ]]; do
  case "$1" in
    --route-a-run-root) ROUTE_A_RUN_ROOT="${2:-}"; shift 2 ;;
    --run-root) RUN_ROOT="${2:-}"; shift 2 ;;
    --methods) METHODS_RAW="${2:-}"; shift 2 ;;
    --partition) PARTITION="${2:-}"; shift 2 ;;
    --test-access-receipt) TEST_ACCESS_RECEIPT="${2:-}"; shift 2 ;;
    --gpu-scribbleprompt) GPU_SP="${2:-}"; shift 2 ;;
    --gpu-nninteractive) GPU_NNI="${2:-}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ -z "${ROUTE_A_RUN_ROOT}" || -z "${RUN_ROOT}" || -z "${METHODS_RAW}" ]]; then
  usage
  exit 2
fi

SELECT_SP=0
SELECT_NNI=0
IFS=',' read -r -a REQUESTED_METHODS <<< "${METHODS_RAW}"
for method in "${REQUESTED_METHODS[@]}"; do
  case "${method}" in
    scribbleprompt)
      if [[ ${SELECT_SP} -eq 1 ]]; then
        echo "Duplicate method in --methods: scribbleprompt" >&2
        exit 2
      fi
      SELECT_SP=1
      ;;
    nninteractive)
      if [[ ${SELECT_NNI} -eq 1 ]]; then
        echo "Duplicate method in --methods: nninteractive" >&2
        exit 2
      fi
      SELECT_NNI=1
      ;;
    *)
      echo "Unsupported method in --methods: ${method:-<empty>}" >&2
      exit 2
      ;;
  esac
done
if [[ ${SELECT_SP} -eq 0 && ${SELECT_NNI} -eq 0 ]]; then
  echo "--methods must select at least one supported comparator" >&2
  exit 2
fi
if [[ "${PARTITION}" != "val" && "${PARTITION}" != "test" ]]; then
  echo "--partition must be val or test" >&2
  exit 2
fi
if [[ "${PARTITION}" == "test" && -z "${TEST_ACCESS_RECEIPT}" ]]; then
  echo "Test access requires --test-access-receipt from the consumed final-freeze grant" >&2
  exit 20
fi
if [[ "${PARTITION}" == "test" && ( ${SELECT_NNI} -ne 1 || ${SELECT_SP} -ne 0 ) ]]; then
  echo "Formal test external execution is admitted only for the frozen nninteractive role" >&2
  exit 20
fi
if [[ "${PARTITION}" == "val" && -n "${TEST_ACCESS_RECEIPT}" ]]; then
  echo "Validation rejects --test-access-receipt" >&2
  exit 20
fi
if [[ "${PARTITION}" == "test" ]]; then
  route_root_resolved="$(readlink -m "${ROUTE_A_RUN_ROOT}")"
  external_root_resolved="$(readlink -m "${RUN_ROOT}")"
  if [[ "${external_root_resolved}" != "${route_root_resolved}/"* ]]; then
    echo "Test external --run-root must be a child of the receipt-bound Route-A root" >&2
    exit 20
  fi
fi
if [[ ${SELECT_SP} -eq 1 && ! "${GPU_SP}" =~ ^[0-9]+$ ]]; then
  echo "--gpu-scribbleprompt must be numeric when ScribblePrompt is selected" >&2
  exit 2
fi
if [[ ${SELECT_NNI} -eq 1 && ! "${GPU_NNI}" =~ ^[0-9]+$ ]]; then
  echo "--gpu-nninteractive must be numeric when nnInteractive is selected" >&2
  exit 2
fi
if [[ ${SELECT_SP} -eq 1 && ${SELECT_NNI} -eq 1 && "${GPU_SP}" == "${GPU_NNI}" ]]; then
  echo "Two selected comparator queues require distinct GPU ids" >&2
  exit 2
fi

CORE_PYTHON_OVERRIDE="${PETCT_CORE_PYTHON:-}"
NNI_PYTHON_OVERRIDE="${PETCT_NNINTERACTIVE_PYTHON:-}"
OFFICIAL_METRICS_OVERRIDE="${PETCT_AUTOPETV_METRICS:-}"
BOOTSTRAP_CORE_PYTHON="${PROJECT_ROOT}/envs/petct_nnunet_v281/bin/python"
CORE_PYTHON="${CORE_PYTHON_OVERRIDE:-${BOOTSTRAP_CORE_PYTHON}}"
SP_PYTHON="${PETCT_SCRIBBLEPROMPT_PYTHON:-${PROJECT_ROOT}/envs/scribbleprompt_v1/bin/python}"
NNI_PYTHON="${NNI_PYTHON_OVERRIDE:-${PROJECT_ROOT}/envs/nninteractive_v1/bin/python}"
SP_SOURCE="${PETCT_SCRIBBLEPROMPT_SOURCE:-${PROJECT_ROOT}/external_runners/scribbleprompt/source}"
SP_CHECKPOINT="${PETCT_SCRIBBLEPROMPT_CHECKPOINT:-${PROJECT_ROOT}/models/ScribblePrompt/ScribblePrompt_unet_v1_nf192_res128.pt}"
NNI_SOURCE="${PETCT_NNINTERACTIVE_SOURCE:-${PROJECT_ROOT}/external_runners/nninteractive/source}"
NNI_MODEL="${PETCT_NNINTERACTIVE_MODEL_FOLDER:-${PROJECT_ROOT}/models/nnInteractive/nnInteractive_v1.0}"
COMPARATOR_CONFIG="${PETCT_EXTERNAL_COMPARATOR_CONFIG:-${PROJECT_ROOT}/configs/petct_external_comparators.json}"
EXPERIMENT_CONFIG="${PETCT_EXPERIMENT_CONFIG:-${PROJECT_ROOT}/configs/petct_route_a_experiment.json}"
OFFICIAL_METRICS="${OFFICIAL_METRICS_OVERRIDE}"
PIPELINE_INPUTS="${ROUTE_A_RUN_ROOT}/artifacts/pipeline_data_inputs.json"
EDITOR_DATA_READY="${ROUTE_A_RUN_ROOT}/artifacts/EDITOR_DATA_READY.json"
SP_ENV_RECEIPT="${PROJECT_ROOT}/records/environments/scribbleprompt_v1.json"
NNI_ENV_RECEIPT="${PROJECT_ROOT}/envs/nninteractive_v1.READY.json"
NNI_ENV_FREEZE="${PROJECT_ROOT}/envs/nninteractive_v1.freeze.txt"
NNI_ADAPTER="${PROJECT_ROOT}/scripts/comparators/nninteractive_petct_adapter.py"
FROZEN_EXTERNAL_ADMISSION=""
FROZEN_TEST_LEARNING_SPLIT=""

# Current admitted adapters expose only a positive/foreground prompt path and
# their derived union-with-M0 policy is ADD-only.  The professor-directed v2
# campaign is bidirectional, so fail before mkdir, data materialization, or GPU
# launch.  REMOVE rows must never be filtered, synthesized, or scored as ADD.
if [[ -f "${EXPERIMENT_CONFIG}" && -f "${COMPARATOR_CONFIG}" ]]; then
  "${CORE_PYTHON}" - "${EXPERIMENT_CONFIG}" "${COMPARATOR_CONFIG}" "${METHODS_RAW}" <<'PY'
import json, sys
experiment = json.load(open(sys.argv[1], encoding="utf-8"))
comparators = json.load(open(sys.argv[2], encoding="utf-8"))
if experiment.get("schema_version") == "PETCT-ROUTE-A-EXPERIMENT-v2.0":
    payload = {
        "schema_version": "PETCT-EXTERNAL-V2-PREFLIGHT-v2.0",
        "status": "REMOVE_UNSUPPORTED",
        "methods": sys.argv[3].split(","),
        "reason": "selected adapters have no admitted negative/background scribble execution contract",
        "union_with_m0_role": "LEGACY_V1_ADD_DERIVED_ONLY",
        "gpu_started": False,
        "output_directory_created": False,
        "comparator_contract_status": comparators.get("status"),
    }
    print(json.dumps(payload, sort_keys=True), file=sys.stderr)
    raise SystemExit(42)
PY
fi

# The formal external gate is resolved from the already-consumed test receipt
# before pipeline_data_inputs, OOF_READY, manifests, predictions, or a GPU are
# touched.  Every runtime/config/source/adapter/model path comes from the same
# receipt-bound final freeze; environment overrides cannot replace them.
if [[ "${PARTITION}" == "test" ]]; then
  export PYTHONNOUSERSITE=1
  for gate_file in "${BOOTSTRAP_CORE_PYTHON}" "${TEST_ACCESS_RECEIPT}" "${EXPERIMENT_CONFIG}" \
    "${PROJECT_ROOT}/scripts/common/petct_development_freeze.py"; do
    if [[ ! -f "${gate_file}" || -L "${gate_file}" ]]; then
      echo "Missing pre-data external-freeze gate prerequisite: ${gate_file}" >&2
      exit 20
    fi
  done
  if [[ ! -x "${BOOTSTRAP_CORE_PYTHON}" ]]; then
    echo "Core Python is not executable for the pre-data external-freeze gate" >&2
    exit 20
  fi
  readarray -t FROZEN_EXTERNAL_FIELDS < <(
    "${BOOTSTRAP_CORE_PYTHON}" -I "${PROJECT_ROOT}/scripts/common/petct_development_freeze.py" \
      resolve-test-external --test-access-receipt "${TEST_ACCESS_RECEIPT}" \
      --experiment-config "${EXPERIMENT_CONFIG}" --run-root "${ROUTE_A_RUN_ROOT}" \
      --method nninteractive |
      "${BOOTSTRAP_CORE_PYTHON}" -I -c '
import json, sys
p = json.load(sys.stdin)
for value in (
    p["admission"]["path"], p["comparator_config"]["path"],
    p["runtime_receipt"]["path"], p["source_root"], p["adapter"]["path"],
    p["environment_freeze"]["path"], p["model_folder"],
    p["model_checkpoint"]["path"], p["model_license"]["path"],
    p["learning_split"]["path"], p["nninteractive_python"]["path"],
    p["core_python"]["path"], p["official_metrics"]["path"],
):
    print(value)
'
  )
  if [[ ${#FROZEN_EXTERNAL_FIELDS[@]} -ne 13 ]]; then
    echo "Could not resolve the receipt-bound nninteractive external role" >&2
    exit 20
  fi
  FROZEN_EXTERNAL_ADMISSION="${FROZEN_EXTERNAL_FIELDS[0]}"
  COMPARATOR_CONFIG="${FROZEN_EXTERNAL_FIELDS[1]}"
  NNI_ENV_RECEIPT="${FROZEN_EXTERNAL_FIELDS[2]}"
  NNI_SOURCE="${FROZEN_EXTERNAL_FIELDS[3]}"
  NNI_ADAPTER="${FROZEN_EXTERNAL_FIELDS[4]}"
  NNI_ENV_FREEZE="${FROZEN_EXTERNAL_FIELDS[5]}"
  NNI_MODEL="${FROZEN_EXTERNAL_FIELDS[6]}"
  FROZEN_NNI_CHECKPOINT="${FROZEN_EXTERNAL_FIELDS[7]}"
  FROZEN_NNI_LICENSE="${FROZEN_EXTERNAL_FIELDS[8]}"
  FROZEN_TEST_LEARNING_SPLIT="${FROZEN_EXTERNAL_FIELDS[9]}"
  RESOLVED_NNI_PYTHON="${FROZEN_EXTERNAL_FIELDS[10]}"
  RESOLVED_CORE_PYTHON="${FROZEN_EXTERNAL_FIELDS[11]}"
  RESOLVED_OFFICIAL_METRICS="${FROZEN_EXTERNAL_FIELDS[12]}"
  if [[ "$(readlink -f -- "${BOOTSTRAP_CORE_PYTHON}")" != "$(readlink -f -- "${RESOLVED_CORE_PYTHON}")" ]]; then
    echo "Isolated bootstrap Python differs from the receipt-bound core Python" >&2
    exit 20
  fi
  for override_pair in \
    "core=${CORE_PYTHON_OVERRIDE}|${RESOLVED_CORE_PYTHON}" \
    "nninteractive=${NNI_PYTHON_OVERRIDE}|${RESOLVED_NNI_PYTHON}" \
    "metrics=${OFFICIAL_METRICS_OVERRIDE}|${RESOLVED_OFFICIAL_METRICS}"; do
    override_name="${override_pair%%=*}"
    override_values="${override_pair#*=}"
    override_path="${override_values%%|*}"
    resolved_path="${override_values#*|}"
    if [[ -n "${override_path}" && "$(readlink -f -- "${override_path}" 2>/dev/null || true)" != "$(readlink -f -- "${resolved_path}")" ]]; then
      echo "Formal test rejects ${override_name} override outside the receipt-bound final freeze" >&2
      exit 20
    fi
  done
  CORE_PYTHON="${RESOLVED_CORE_PYTHON}"
  NNI_PYTHON="${RESOLVED_NNI_PYTHON}"
  OFFICIAL_METRICS="${RESOLVED_OFFICIAL_METRICS}"
  if [[ "$(readlink -m "${NNI_MODEL}/fold_0/checkpoint_final.pth")" != "$(readlink -m "${FROZEN_NNI_CHECKPOINT}")" \
    || "$(readlink -m "${NNI_MODEL}/LICENSE")" != "$(readlink -m "${FROZEN_NNI_LICENSE}")" ]]; then
    echo "Resolved nninteractive model folder differs from its frozen checkpoint/license" >&2
    exit 20
  fi
fi

declare -a REQUIRED_FILES=(
  "${CORE_PYTHON}"
  "${COMPARATOR_CONFIG}"
  "${EXPERIMENT_CONFIG}"
  "${OFFICIAL_METRICS}"
  "${PIPELINE_INPUTS}"
  "${EDITOR_DATA_READY}"
  "${PROJECT_ROOT}/scripts/baseline/validate_petct_m0_oof.py"
  "${PROJECT_ROOT}/scripts/common/petct_test_access.py"
  "${PROJECT_ROOT}/scripts/comparators/build_petct_external_comparator_manifest.py"
  "${PROJECT_ROOT}/scripts/comparators/evaluate_petct_external_comparator.py"
  "${PROJECT_ROOT}/scripts/comparators/finalize_petct_external_comparators.py"
  "${PROJECT_ROOT}/scripts/orchestration/validate_petct_route_a_receipt_pipeline.py"
)
if [[ "${PARTITION}" == "test" ]]; then
  REQUIRED_FILES+=("${TEST_ACCESS_RECEIPT}")
fi
declare -a SELECTED_PYTHONS=("${CORE_PYTHON}")
if [[ ${SELECT_SP} -eq 1 ]]; then
  REQUIRED_FILES+=(
    "${SP_PYTHON}"
    "${SP_ENV_RECEIPT}"
    "${SP_CHECKPOINT}"
    "${PROJECT_ROOT}/scripts/comparators/scribbleprompt_petct_adapter.py"
  )
  SELECTED_PYTHONS+=("${SP_PYTHON}")
fi
if [[ ${SELECT_NNI} -eq 1 ]]; then
  REQUIRED_FILES+=(
    "${NNI_PYTHON}"
    "${NNI_ENV_RECEIPT}"
    "${NNI_ENV_FREEZE}"
    "${NNI_SOURCE}/LICENSE"
    "${NNI_MODEL}/LICENSE"
    "${NNI_MODEL}/fold_0/checkpoint_final.pth"
    "${NNI_ADAPTER}"
  )
  SELECTED_PYTHONS+=("${NNI_PYTHON}")
fi
for required in "${REQUIRED_FILES[@]}"; do
  if [[ -z "${required}" || ! -f "${required}" || -L "${required}" ]]; then
    echo "Missing regular prerequisite: ${required:-<unset>}" >&2
    exit 3
  fi
done
for executable in "${SELECTED_PYTHONS[@]}"; do
  if [[ ! -x "${executable}" ]]; then
    echo "Python is not executable: ${executable}" >&2
    exit 3
  fi
done
for ((i = 0; i < ${#SELECTED_PYTHONS[@]}; i++)); do
  for ((j = i + 1; j < ${#SELECTED_PYTHONS[@]}; j++)); do
    if [[ "$(readlink -m "${SELECTED_PYTHONS[i]}")" == "$(readlink -m "${SELECTED_PYTHONS[j]}")" ]]; then
      echo "Selected comparators and core must use independent Python environments" >&2
      exit 3
    fi
  done
done
if [[ ${SELECT_SP} -eq 1 && ( ! -d "${SP_SOURCE}" || -L "${SP_SOURCE}" ) ]]; then
  echo "Pinned ScribblePrompt source directory is unavailable" >&2
  exit 3
fi
if [[ ${SELECT_NNI} -eq 1 && ( ! -d "${NNI_SOURCE}" || -L "${NNI_SOURCE}" ) ]]; then
  echo "Pinned nnInteractive source directory is unavailable" >&2
  exit 3
fi
if [[ ${SELECT_NNI} -eq 1 && ( ! -d "${NNI_MODEL}" || -L "${NNI_MODEL}" ) ]]; then
  echo "Pinned nnInteractive model directory is unavailable" >&2
  exit 3
fi
if [[ -e "${RUN_ROOT}" || -L "${RUN_ROOT}" ]]; then
  echo "Run root already exists; refusing overwrite: ${RUN_ROOT}" >&2
  exit 4
fi

# ARGV_WIRED is only an explicit wiring gate. Runtime admission is verified
# independently below; reference-only and not-wired methods remain unreachable.
"${CORE_PYTHON}" - \
  "${COMPARATOR_CONFIG}" "${METHODS_RAW}" "${SP_ENV_RECEIPT}" "${NNI_ENV_RECEIPT}" \
  "${NNI_SOURCE}" "${NNI_MODEL}" \
  "${NNI_ADAPTER}" "${NNI_ENV_FREEZE}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_bundle_sha256(root):
    root = Path(root)
    records = []
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        if not path.is_file() or path.is_symlink():
            continue
        rel = path.relative_to(root)
        if any(part == "__pycache__" or part.endswith(".egg-info") for part in rel.parts):
            continue
        if path.suffix in {".pyc", ".pyo"}:
            continue
        records.append(f"{sha256(path)}  {rel.as_posix()}")
    return hashlib.sha256(("\n".join(records) + "\n").encode("utf-8")).hexdigest(), len(records)


config_path = Path(sys.argv[1])
project_root = config_path.resolve().parent.parent
sys.path.insert(0, str(project_root / "scripts" / "comparators"))
from run_petct_external_comparator import (
    load_and_validate_contract,
    validate_execution_admission,
)

config = load_and_validate_contract(config_path)
selected = sys.argv[2].split(",")
receipt_paths = {"scribbleprompt": Path(sys.argv[3]), "nninteractive": Path(sys.argv[4])}
nni_source = Path(sys.argv[5])
nni_model = Path(sys.argv[6])
nni_adapter = Path(sys.argv[7])
nni_environment_freeze = Path(sys.argv[8])
methods = {method["id"]: method for method in config.get("methods", [])}
expected = {
    "scribbleprompt": "RUN_FORMAL_2D_SPATIAL",
    "nninteractive": "RUN_SECONDARY_EXPOSED_PRETRAINING",
}
for method_id in selected:
    selection = expected[method_id]
    method = methods.get(method_id)
    if not isinstance(method, dict) or method.get("selection") != selection:
        raise SystemExit(f"method selection gate failed: {method_id}")
    execution = method.get("execution", {})
    if execution.get("state") != "ARGV_WIRED":
        raise SystemExit(f"method argv is not wired: {method_id}")
    if execution.get("network_policy") != "NO_DOWNLOADS":
        raise SystemExit(f"method lost its NO_DOWNLOADS gate: {method_id}")
    validate_execution_admission(
        method,
        config_path,
        variables={"project_root": str(project_root)},
    )
    receipt = json.load(receipt_paths[method_id].open(encoding="utf-8"))
    if method_id == "scribbleprompt":
        valid = receipt.get("schema_version") == "PETCT-SCRIBBLEPROMPT-ENV-v1.0" and receipt.get("official_unet_loaded_on_cpu")
    else:
        source_hash, source_count = source_bundle_sha256(nni_source)
        availability = method.get("pretraining", {}).get("local_checkpoint_availability", {})
        expected_metadata = {
            name: sha256(nni_model / name)
            for name in ("dataset.json", "plans.json", "inference_session_class.json")
        }
        valid = all(
            (
                receipt.get("schema_version") == "PETCT-NNINTERACTIVE-ENV-v1.1",
                receipt.get("status") == "PASS",
                receipt.get("cuda_available") is True,
                str(receipt.get("smoke_device", "")).startswith("cuda"),
                receipt.get("source_commit") == method.get("source", {}).get("pinned_commit"),
                receipt.get("source_bundle_sha256") == source_hash,
                receipt.get("source_bundle_file_count") == source_count,
                receipt.get("source_license_sha256") == sha256(nni_source / "LICENSE"),
                receipt.get("checkpoint_sha256") == availability.get("sha256") == sha256(nni_model / "fold_0/checkpoint_final.pth"),
                receipt.get("license_sha256") == availability.get("license_sha256") == sha256(nni_model / "LICENSE"),
                receipt.get("license") == availability.get("license") == "CC BY-NC-SA 4.0",
                receipt.get("model_metadata_sha256") == expected_metadata,
                receipt.get("config_sha256") == sha256(config_path),
                receipt.get("adapter_sha256") == sha256(nni_adapter),
                receipt.get("environment_freeze_sha256") == sha256(nni_environment_freeze),
                receipt.get("model_load_smoke") == "PASS",
                receipt.get("initial_m0_api_smoke") == "PASS",
                receipt.get("scribble_api_smoke") == "PASS",
                receipt.get("adapter_cli_smoke") == "PASS",
                receipt.get("synthetic_only") is True,
                receipt.get("scientific_prediction_produced") is False,
                receipt.get("network_policy_at_runtime") == "NO_DOWNLOADS",
            )
        )
    if not valid:
        raise SystemExit(f"environment receipt is invalid: {method_id}")
print(json.dumps({"status": "READY", "methods": sorted(selected)}))
PY

readarray -t DATA_PATHS < <("${CORE_PYTHON}" - "${PIPELINE_INPUTS}" <<'PY'
import json, sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
for key in ("case_manifest", "learning_split", "oof_ready", "natural_episode_manifest", "natural_tensor_manifest"):
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise SystemExit(f"pipeline_data_inputs is missing {key}")
    print(value)
PY
)
if [[ ${#DATA_PATHS[@]} -ne 5 ]]; then
  echo "Could not resolve the frozen Route-A data receipts" >&2
  exit 5
fi
CASE_MANIFEST="${DATA_PATHS[0]}"
LEARNING_SPLIT="${DATA_PATHS[1]}"
OOF_READY="${DATA_PATHS[2]}"
NATURAL_EPISODES="${DATA_PATHS[3]}"
NATURAL_TENSORS="${DATA_PATHS[4]}"
if [[ "${PARTITION}" == "test" && "$(readlink -m "${LEARNING_SPLIT}")" != "$(readlink -m "${FROZEN_TEST_LEARNING_SPLIT}")" ]]; then
  echo "Route-A pipeline learning split differs from the consumed external freeze" >&2
  exit 20
fi
for required in "${CASE_MANIFEST}" "${LEARNING_SPLIT}" "${OOF_READY}" "${NATURAL_EPISODES}" "${NATURAL_TENSORS}"; do
  if [[ ! -f "${required}" || -L "${required}" ]]; then
    echo "Frozen Route-A input is missing: ${required}" >&2
    exit 5
  fi
done

# This validation is intentionally before mkdir and before either adapter. It
# makes it impossible for this launcher to start a GPU method before OOF_READY.
"${CORE_PYTHON}" "${PROJECT_ROOT}/scripts/baseline/validate_petct_m0_oof.py" \
  validate-ready-receipt "${OOF_READY}" >/dev/null

# This launcher only validates an already-consumed grant.  It never creates or
# consumes test authorization.  Validation therefore remains exactly-once and
# owned by the primary Route-A run.
if [[ "${PARTITION}" == "test" ]]; then
  "${CORE_PYTHON}" "${PROJECT_ROOT}/scripts/common/petct_test_access.py" validate \
    --receipt "${TEST_ACCESS_RECEIPT}" --experiment-config "${EXPERIMENT_CONFIG}" \
    --learning-split "${LEARNING_SPLIT}" --run-root "${ROUTE_A_RUN_ROOT}" >/dev/null
fi

mkdir -p "${RUN_ROOT}/logs" "${RUN_ROOT}/state" "${RUN_ROOT}/artifacts" \
  "${RUN_ROOT}/predictions" "${RUN_ROOT}/metrics"

run_stage() {
  local name="$1"
  shift
  local running="${RUN_ROOT}/state/${name}.running"
  local done="${RUN_ROOT}/state/${name}.done"
  local failed="${RUN_ROOT}/state/${name}.fail"
  for marker in "${running}" "${done}" "${failed}"; do
    if [[ -e "${marker}" ]]; then
      echo "Refusing existing stage marker: ${marker}" >&2
      return 90
    fi
  done
  printf 'stage=%s\nstarted_at=%s\n' "${name}" "$(date --iso-8601=seconds)" > "${running}"
  set +e
  "$@" > >(tee "${RUN_ROOT}/logs/${name}.log") 2>&1
  local status=$?
  set -e
  printf 'finished_at=%s\nexit_code=%s\n' "$(date --iso-8601=seconds)" "${status}" >> "${running}"
  if [[ ${status} -eq 0 ]]; then
    mv "${running}" "${done}"
  else
    mv "${running}" "${failed}"
  fi
  return "${status}"
}

run_stage data_receipt_gate \
  "${CORE_PYTHON}" "${PROJECT_ROOT}/scripts/orchestration/validate_petct_route_a_receipt_pipeline.py" \
  --inputs "${PIPELINE_INPUTS}" --target editor_data \
  --output "${RUN_ROOT}/artifacts/EXTERNAL_DATA_GATE.json"

INPUT_MANIFEST="${RUN_ROOT}/artifacts/external_comparator_input.json"
declare -a TEST_ACCESS_ARGS=()
if [[ "${PARTITION}" == "test" ]]; then
  TEST_ACCESS_ARGS+=(
    --test-access-receipt "${TEST_ACCESS_RECEIPT}"
    --run-root "${ROUTE_A_RUN_ROOT}"
  )
fi
run_stage build_external_input \
  "${CORE_PYTHON}" "${PROJECT_ROOT}/scripts/comparators/build_petct_external_comparator_manifest.py" \
  --oof-ready "${OOF_READY}" --case-manifest "${CASE_MANIFEST}" \
  --learning-split "${LEARNING_SPLIT}" --natural-episode-manifest "${NATURAL_EPISODES}" \
  --natural-tensor-manifest "${NATURAL_TENSORS}" --experiment-config "${EXPERIMENT_CONFIG}" \
  --partition "${PARTITION}" --scribble-dir "${RUN_ROOT}/artifacts/frozen_scribbles" \
  --output-manifest "${INPUT_MANIFEST}" "${TEST_ACCESS_ARGS[@]}"

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=offline PIP_NO_INDEX=1
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

evaluate_one() {
  local method="$1"
  local policy="$2"
  local output_manifest="$3"
  run_stage "evaluate_${method}_${policy}" \
    "${CORE_PYTHON}" "${PROJECT_ROOT}/scripts/comparators/evaluate_petct_external_comparator.py" \
    --input-manifest "${INPUT_MANIFEST}" --output-manifest "${output_manifest}" \
    --natural-episode-manifest "${NATURAL_EPISODES}" --comparator-config "${COMPARATOR_CONFIG}" \
    --experiment-config "${EXPERIMENT_CONFIG}" --learning-split "${LEARNING_SPLIT}" \
    --partition "${PARTITION}" --official-metrics "${OFFICIAL_METRICS}" \
    --method "${method}" --output-policy "${policy}" \
    --rows "${RUN_ROOT}/metrics/${method}_${policy}_rows.jsonl" \
    --summary "${RUN_ROOT}/metrics/${method}_${policy}_summary.json" "${TEST_ACCESS_ARGS[@]}"
}

run_scribbleprompt_queue() {
  local native_manifest native_dir union_manifest union_dir
  native_manifest="${RUN_ROOT}/artifacts/scribbleprompt_native_slice_replace_output.json"
  native_dir="${RUN_ROOT}/predictions/scribbleprompt_native_slice_replace"
  union_manifest="${RUN_ROOT}/artifacts/scribbleprompt_union_with_m0_output.json"
  union_dir="${RUN_ROOT}/predictions/scribbleprompt_union_with_m0"
  # One official model call per record. The adapter writes the native output and
  # deterministically derives the add-only policy from that exact prediction.
  run_stage "infer_scribbleprompt_once" env CUDA_VISIBLE_DEVICES="${GPU_SP}" \
    "${SP_PYTHON}" "${PROJECT_ROOT}/scripts/comparators/scribbleprompt_petct_adapter.py" \
    --input-manifest "${INPUT_MANIFEST}" --output-manifest "${native_manifest}" \
    --output-dir "${native_dir}" --checkpoint "${SP_CHECKPOINT}" --source-root "${SP_SOURCE}" \
    --output-policy native_slice_replace \
    --derived-union-output-manifest "${union_manifest}" \
    --derived-union-output-dir "${union_dir}" \
    --device cuda --partition "${PARTITION}" \
    --learning-split "${LEARNING_SPLIT}" --experiment-config "${EXPERIMENT_CONFIG}" \
    "${TEST_ACCESS_ARGS[@]}"
  # Keep the positive-only diagnostic table first; both tables share one native inference.
  evaluate_one scribbleprompt union_with_m0 "${union_manifest}"
  evaluate_one scribbleprompt native_slice_replace "${native_manifest}"
}

run_nninteractive_queue() {
  local native_manifest native_dir union_manifest union_dir
  native_manifest="${RUN_ROOT}/artifacts/nninteractive_native_full_mask_output.json"
  native_dir="${RUN_ROOT}/predictions/nninteractive_native_full_mask"
  union_manifest="${RUN_ROOT}/artifacts/nninteractive_union_with_m0_output.json"
  union_dir="${RUN_ROOT}/predictions/nninteractive_union_with_m0"
  # M0 initialization does not predict; the one foreground scribble triggers
  # exactly one native prediction, from which the add-only output is derived.
  run_stage "infer_nninteractive_once" env CUDA_VISIBLE_DEVICES="${GPU_NNI}" \
    "${NNI_PYTHON}" "${NNI_ADAPTER}" \
    --input-manifest "${INPUT_MANIFEST}" --output-manifest "${native_manifest}" \
    --output-dir "${native_dir}" --model-folder "${NNI_MODEL}" \
    --config "${COMPARATOR_CONFIG}" --output-policy native_full_mask \
    --derived-union-output-manifest "${union_manifest}" \
    --derived-union-output-dir "${union_dir}" \
    --device cuda:0 --torch-threads 4 --partition "${PARTITION}" \
    --learning-split "${LEARNING_SPLIT}" --experiment-config "${EXPERIMENT_CONFIG}" \
    "${TEST_ACCESS_ARGS[@]}"
  evaluate_one nninteractive union_with_m0 "${union_manifest}"
  evaluate_one nninteractive native_full_mask "${native_manifest}"
}

declare -a QUEUE_NAMES=()
declare -a QUEUE_PIDS=()
if [[ ${SELECT_SP} -eq 1 ]]; then
  run_scribbleprompt_queue &
  QUEUE_NAMES+=("scribbleprompt")
  QUEUE_PIDS+=("$!")
fi
if [[ ${SELECT_NNI} -eq 1 ]]; then
  run_nninteractive_queue &
  QUEUE_NAMES+=("nninteractive")
  QUEUE_PIDS+=("$!")
fi

set +e
QUEUE_FAILURES=()
for i in "${!QUEUE_PIDS[@]}"; do
  wait "${QUEUE_PIDS[i]}"
  status=$?
  if [[ ${status} -ne 0 ]]; then
    QUEUE_FAILURES+=("${QUEUE_NAMES[i]}=${status}")
  fi
done
set -e
if [[ ${#QUEUE_FAILURES[@]} -ne 0 ]]; then
  echo "External comparator queues failed: ${QUEUE_FAILURES[*]}" >&2
  exit 30
fi

# This remains an inventory receipt, not a pooled result table.  v1.2 binds
# config, experiment/split, runtime receipt, input/output manifests, JSONL rows,
# summaries, and every prediction leaf.  Formal test additionally binds the
# already-frozen optional nnInteractive admission and verifies actual checkpoint use.
declare -a COMPLETE_ARGS=(
  build --run-root "${RUN_ROOT}" --partition "${PARTITION}"
  --comparator-config "${COMPARATOR_CONFIG}" --experiment-config "${EXPERIMENT_CONFIG}"
  --learning-split "${LEARNING_SPLIT}" --input-manifest "${INPUT_MANIFEST}"
  --natural-episode-manifest "${NATURAL_EPISODES}"
  --core-python "${CORE_PYTHON}" --official-metrics "${OFFICIAL_METRICS}"
  --output "${RUN_ROOT}/artifacts/EXTERNAL_COMPARATORS_COMPLETE.json"
)
if [[ ${SELECT_SP} -eq 1 ]]; then
  COMPLETE_ARGS+=(--method scribbleprompt --runtime-receipt "scribbleprompt=${SP_ENV_RECEIPT}")
fi
if [[ ${SELECT_NNI} -eq 1 ]]; then
  COMPLETE_ARGS+=(
    --method nninteractive --runtime-receipt "nninteractive=${NNI_ENV_RECEIPT}"
    --nninteractive-python "${NNI_PYTHON}"
  )
fi
if [[ "${PARTITION}" == "test" ]]; then
  COMPLETE_ARGS+=(
    --test-access-receipt "${TEST_ACCESS_RECEIPT}"
    --frozen-external-admission "${FROZEN_EXTERNAL_ADMISSION}"
    --route-a-run-root "${ROUTE_A_RUN_ROOT}"
  )
fi
run_stage validate_external_complete \
  "${CORE_PYTHON}" "${PROJECT_ROOT}/scripts/comparators/finalize_petct_external_comparators.py" \
  "${COMPLETE_ARGS[@]}"

# Validation is where a new optional external admission may be created.  It is
# later supplied to the final-development-freeze builder.  Test never rebuilds
# or replaces this object.
if [[ "${PARTITION}" == "val" && ${SELECT_NNI} -eq 1 ]]; then
  run_stage freeze_nninteractive_external_admission \
    "${CORE_PYTHON}" "${PROJECT_ROOT}/scripts/common/petct_development_freeze.py" \
    build-external-admission --comparator-config "${COMPARATOR_CONFIG}" \
    --experiment-config "${EXPERIMENT_CONFIG}" --learning-split "${LEARNING_SPLIT}" \
    --validation-complete "${RUN_ROOT}/artifacts/EXTERNAL_COMPARATORS_COMPLETE.json" \
    --output "${RUN_ROOT}/artifacts/NNINTERACTIVE_EXTERNAL_ADMISSION.json"
fi

echo "External comparators complete: ${RUN_ROOT}/artifacts/EXTERNAL_COMPARATORS_COMPLETE.json"
