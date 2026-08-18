#!/usr/bin/env bash
set -euo pipefail

echo "HISTORICAL_ONLY: superseded Route A launcher cannot run after the R13/M0-v6 cutover." >&2
exit 64

# Run the implemented Route A stages after the five-fold nnU-Net baseline.
# The launcher is deliberately fail-closed.  Controlled matched-state tensors
# train P2T; the resulting checkpoint is then applied to natural OOF tensors by
# a receipt-preserving cross-manifest inference CLI before predicted_slots.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${PROJECT_ROOT}/scripts/common/petct_m0_common.sh"

usage() {
  cat >&2 <<'EOF'
Usage:
  run_petct_route_a_after_baseline.sh --run-root DIR [options]

Options:
  --partition val|test       Evaluation partition (default: val).
  --final-freeze-grant FILE  Externally pre-created grant; required for test.
  --gpu0 ID                  First CUDA device (default: 0).
  --gpu1 ID                  Second CUDA device (default: 1).

Required environment:
  PETCT_PSMA_AUDIT_JSON      PASS PSMA-v3 audit JSON.
  PETCT_AUTOPETV_SIMULATOR  Pinned AutoPET V scribble simulator Python file.
  PETCT_AUTOPETV_METRICS    Pinned AutoPET V lesion-metrics Python file.
  PETCT_AUTOPETV_RUNTIME_MANIFEST  Frozen minimal-runtime manifest (optional;
                                  defaults to protocols/autopetv_protocol_runtime.json).

Optional environment:
  PETCT_EXPERIMENT_CONFIG    Defaults to configs/petct_route_a_experiment.json.
  PETCT_SOURCE_DATASET       Defaults to the source dataset from petct_m0_common.sh.
EOF
}

RUN_ROOT=""
EVALUATION_PARTITION="val"
FINAL_FREEZE_GRANT=""
GPU0_ID=0
GPU1_ID=1
while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-root) RUN_ROOT="${2:-}"; shift 2 ;;
    --partition) EVALUATION_PARTITION="${2:-}"; shift 2 ;;
    --final-freeze-grant) FINAL_FREEZE_GRANT="${2:-}"; shift 2 ;;
    --gpu0) GPU0_ID="${2:-}"; shift 2 ;;
    --gpu1) GPU1_ID="${2:-}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ -z "${RUN_ROOT}" || ! "${GPU0_ID}" =~ ^[0-9]+$ || ! "${GPU1_ID}" =~ ^[0-9]+$ ]]; then
  usage
  exit 2
fi
if [[ "${GPU0_ID}" == "${GPU1_ID}" ]]; then
  echo "Two distinct GPU ids are required." >&2
  exit 2
fi
if [[ "${EVALUATION_PARTITION}" != "val" && "${EVALUATION_PARTITION}" != "test" ]]; then
  echo "--partition must be val or test." >&2
  exit 2
fi
if [[ "${EVALUATION_PARTITION}" == "test" && -z "${FINAL_FREEZE_GRANT}" ]]; then
  echo "Test access is locked; pass an externally pre-created --final-freeze-grant." >&2
  exit 20
fi
if [[ "${EVALUATION_PARTITION}" == "val" && -n "${FINAL_FREEZE_GRANT}" ]]; then
  echo "--final-freeze-grant is valid only together with --partition test." >&2
  exit 20
fi

EXPERIMENT_CONFIG="${PETCT_EXPERIMENT_CONFIG:-${PROJECT_ROOT}/configs/petct_route_a_experiment.json}"
SOURCE_DATASET="${PETCT_SOURCE_DATASET:-${SOURCE_DATASET}}"
AUDIT_JSON="${PETCT_PSMA_AUDIT_JSON:-}"
OFFICIAL_SIMULATOR="${PETCT_AUTOPETV_SIMULATOR:-}"
OFFICIAL_METRICS="${PETCT_AUTOPETV_METRICS:-}"
OFFICIAL_RUNTIME_MANIFEST="${PETCT_AUTOPETV_RUNTIME_MANIFEST:-${PROJECT_ROOT}/protocols/autopetv_protocol_runtime.json}"
CORE_ENV_RECEIPT="${EXP_ROOT}/envs/petct_nnunet_v281.json"
CORE_ENV_MARKER="${EXP_ROOT}/envs/ENV_READY.done"
FULL_TRAIN_READY="${EXP_ROOT}/manifests/FULL_TRAIN_READY.json"
OOF_READY="${EXP_ROOT}/manifests/OOF_READY.json"
F0_VALIDATOR="${PROJECT_ROOT}/scripts/orchestration/validate_petct_route_a_f0.py"
F0_READY="${PROJECT_ROOT}/route_a/manifests/F0_READY.json"
EXPECTED_F0_ENV_BUNDLE="87a2261af9d99eb8232a078a2f7ba81cf9f3b4a6389410c296ca9b8671246006"
FROZEN_M0_VALIDATION_RECEIPT=""
FROZEN_OOF_RECEIPT=""
FROZEN_ENVIRONMENT_RECEIPT=""

declare -a REQUIRED_FILES=(
  "${EXPERIMENT_CONFIG}"
  "${AUDIT_JSON}"
  "${OFFICIAL_SIMULATOR}"
  "${OFFICIAL_METRICS}"
  "${OFFICIAL_RUNTIME_MANIFEST}"
  "${CORE_ENV_MARKER}"
  "${F0_VALIDATOR}"
  "${F0_READY}"
  "${PROJECT_ROOT}/scripts/baseline/run_petct_m0_oof_parallel.sh"
  "${PROJECT_ROOT}/scripts/baseline/validate_petct_m0_oof.py"
  "${PROJECT_ROOT}/scripts/data/build_petct_source_case_manifest.py"
  "${PROJECT_ROOT}/scripts/evaluation/evaluate_petct_m0_oof.py"
  "${PROJECT_ROOT}/scripts/data/build_petct_learning_split.py"
  "${PROJECT_ROOT}/scripts/data/validate_petct_learning_split.py"
  "${PROJECT_ROOT}/scripts/data/build_petct_residual_manifest.py"
  "${PROJECT_ROOT}/scripts/p2t/build_petct_matched_state_dataset.py"
  "${PROJECT_ROOT}/scripts/data/build_petct_scribble_dataset.py"
  "${PROJECT_ROOT}/scripts/data/materialize_petct_learning_tensors.py"
  "${PROJECT_ROOT}/scripts/p2t/train_petct_p2t.py"
  "${PROJECT_ROOT}/scripts/p2t/infer_petct_p2t.py"
  "${PROJECT_ROOT}/scripts/evaluation/evaluate_petct_p2t.py"
  "${PROJECT_ROOT}/scripts/editor/train_petct_residual_editor.py"
  "${PROJECT_ROOT}/scripts/editor/build_petct_intent_interventions.py"
  "${PROJECT_ROOT}/scripts/editor/infer_petct_residual_editor.py"
  "${PROJECT_ROOT}/scripts/evaluation/evaluate_petct_correction.py"
  "${PROJECT_ROOT}/scripts/evaluation/aggregate_petct_condition_metrics.py"
  "${PROJECT_ROOT}/scripts/evaluation/aggregate_petct_p2t_confirmatory.py"
  "${PROJECT_ROOT}/scripts/common/petct_test_access.py"
  "${PROJECT_ROOT}/scripts/common/petct_development_freeze.py"
  "${PROJECT_ROOT}/scripts/orchestration/validate_petct_route_a_receipt_pipeline.py"
)
if [[ "${EVALUATION_PARTITION}" == "test" ]]; then
  REQUIRED_FILES+=("${FINAL_FREEZE_GRANT}")
else
  REQUIRED_FILES+=("${CORE_ENV_RECEIPT}" "${FULL_TRAIN_READY}")
fi
for required in "${REQUIRED_FILES[@]}"; do
  if [[ -z "${required}" || ! -f "${required}" || -L "${required}" ]]; then
    echo "Missing regular prerequisite: ${required:-<unset environment path>}" >&2
    exit 3
  fi
done
if [[ ! -d "${SOURCE_DATASET}" || -L "${SOURCE_DATASET}" || ! -x "${PYTHON}" ]]; then
  echo "Source dataset or pinned Python is unavailable." >&2
  exit 3
fi
if [[ -e "${RUN_ROOT}" || -L "${RUN_ROOT}" ]]; then
  echo "Run root already exists; refusing overwrite: ${RUN_ROOT}" >&2
  exit 4
fi

# F0 is the one pre-run technical-readiness gate for the seven closed
# cross-stage blockers.  Revalidate the immutable receipt against the exact
# deployed scripts, canonical inputs, and core environment before creating a
# run root.  This keeps a stale or missing F0 failure safely retryable.
F0_VALIDATION_JSON="$("${PYTHON}" "${F0_VALIDATOR}" validate \
  --receipt "${F0_READY}" --project-root "${PROJECT_ROOT}" \
  --experiment-config "${EXPERIMENT_CONFIG}" \
  --environment-marker "${CORE_ENV_MARKER}" \
  --official-simulator "${OFFICIAL_SIMULATOR}" \
  --official-metrics "${OFFICIAL_METRICS}" \
  --official-runtime-manifest "${OFFICIAL_RUNTIME_MANIFEST}" \
  --expected-env-bundle "${EXPECTED_F0_ENV_BUNDLE}")"
readarray -t F0_RECEIPT_RECORD < <("${PYTHON}" -c '
import json, sys
record = json.loads(sys.argv[1])["receipt"]
print(record["path"])
print(record["sha256"])
print(record["bytes"])
' "${F0_VALIDATION_JSON}")
if [[ ${#F0_RECEIPT_RECORD[@]} -ne 3 ]]; then
  echo "F0 validator returned an incomplete receipt record." >&2
  exit 4
fi
F0_READY_PATH="${F0_RECEIPT_RECORD[0]}"
F0_READY_SHA256="${F0_RECEIPT_RECORD[1]}"
F0_READY_BYTES="${F0_RECEIPT_RECORD[2]}"

mkdir -p "${RUN_ROOT}/logs" "${RUN_ROOT}/state" "${RUN_ROOT}/artifacts" \
  "${RUN_ROOT}/metrics/p2t_natural" "${RUN_ROOT}/models/p2t" \
  "${RUN_ROOT}/models/editor" "${RUN_ROOT}/governance"

run_stage() {
  local name="$1"
  shift
  local running="${RUN_ROOT}/state/${name}.running"
  local done="${RUN_ROOT}/state/${name}.done"
  local failed="${RUN_ROOT}/state/${name}.fail"
  if [[ -e "${running}" || -e "${done}" || -e "${failed}" ]]; then
    echo "Refusing existing stage marker for ${name}." >&2
    return 90
  fi
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

# Test authorization is consumed exactly once, immediately after RUN_ROOT is
# created and before any OOF receipt, source manifest, or experiment data is
# opened.  Every downstream leaf and pipeline receipt revalidates this same
# immutable consumed receipt.
TEST_ACCESS_RECEIPT=""
FINAL_DEVELOPMENT_FREEZE=""
FROZEN_CHECKPOINT_BINDINGS=""
LEARNING_SPLIT=""
declare -a TEST_ACCESS_ARGS=()
if [[ "${EVALUATION_PARTITION}" == "test" ]]; then
  LEARNING_SPLIT="$("${PYTHON}" -c '
import json, sys
from pathlib import Path
grant = json.load(open(sys.argv[1], encoding="utf-8"))
raw = Path(grant.get("learning_split", {}).get("path", ""))
if raw.is_symlink() or not raw.is_file():
    raise SystemExit("final-freeze grant does not bind a non-symlink regular learning split")
print(raw.resolve())
' "${FINAL_FREEZE_GRANT}")"
  TEST_ACCESS_RECEIPT="${RUN_ROOT}/governance/TEST_ACCESS_CONSUMED.json"
  run_stage consume_test_access \
    "${PYTHON}" "${PROJECT_ROOT}/scripts/common/petct_test_access.py" consume \
    --grant "${FINAL_FREEZE_GRANT}" --run-root "${RUN_ROOT}" \
    --receipt "${TEST_ACCESS_RECEIPT}"
  TEST_ACCESS_ARGS+=(
    --test-access-receipt "${TEST_ACCESS_RECEIPT}"
    --run-root "${RUN_ROOT}"
  )
  FINAL_DEVELOPMENT_FREEZE="$("${PYTHON}" -c '
import json, sys
receipt = json.load(open(sys.argv[1], encoding="utf-8"))
print(receipt["consumption"]["final_development_freeze"]["path"])
' "${TEST_ACCESS_RECEIPT}")"
  FROZEN_CHECKPOINT_BINDINGS="${RUN_ROOT}/governance/FROZEN_CHECKPOINT_BINDINGS.json"
  run_stage export_frozen_checkpoint_bindings \
    "${PYTHON}" "${PROJECT_ROOT}/scripts/common/petct_development_freeze.py" \
    export-checkpoints --freeze "${FINAL_DEVELOPMENT_FREEZE}" \
    --experiment-config "${EXPERIMENT_CONFIG}" --learning-split "${LEARNING_SPLIT}" \
    --output "${FROZEN_CHECKPOINT_BINDINGS}"
  FROZEN_ENVIRONMENT_RECEIPT="$("${PYTHON}" \
    "${PROJECT_ROOT}/scripts/common/petct_development_freeze.py" resolve-artifact \
    --bindings "${FROZEN_CHECKPOINT_BINDINGS}" --role environment_receipt)"
  CORE_ENV_RECEIPT="$("${PYTHON}" -c '
import hashlib, json, sys
marker = json.load(open(sys.argv[1], encoding="utf-8"))
path = marker.get("receipt_path", "")
with open(path, "rb") as stream:
    observed = hashlib.sha256(stream.read()).hexdigest()
if observed != marker.get("receipt_sha256"):
    raise SystemExit("freeze-bound environment receipt hash mismatch")
print(path)
' "${FROZEN_ENVIRONMENT_RECEIPT}")"
  OOF_READY="$("${PYTHON}" \
    "${PROJECT_ROOT}/scripts/common/petct_development_freeze.py" resolve-artifact \
    --bindings "${FROZEN_CHECKPOINT_BINDINGS}" --role m0_oof_receipt)"
  FROZEN_M0_VALIDATION_RECEIPT="$("${PYTHON}" \
    "${PROJECT_ROOT}/scripts/common/petct_development_freeze.py" resolve-artifact \
    --bindings "${FROZEN_CHECKPOINT_BINDINGS}" --role m0_validation_receipt)"
  FROZEN_OOF_RECEIPT="${OOF_READY}"
  for frozen_prerequisite in \
    "${FROZEN_ENVIRONMENT_RECEIPT}" "${CORE_ENV_RECEIPT}" "${OOF_READY}" \
    "${FROZEN_M0_VALIDATION_RECEIPT}"; do
    if [[ ! -f "${frozen_prerequisite}" || -L "${frozen_prerequisite}" ]]; then
      echo "Frozen formal-test prerequisite is missing or a symlink: ${frozen_prerequisite}" >&2
      exit 21
    fi
  done
fi

run_stage environment_preflight "${PYTHON}" -c '
import hashlib, json, sys
from pathlib import Path
receipt_path, prefix_raw, simulator_raw, metrics_raw = map(Path, sys.argv[1:])
receipt = json.load(receipt_path.open(encoding="utf-8"))
prefix = prefix_raw.resolve()
simulator = simulator_raw.resolve()
metrics = metrics_raw.resolve()
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
if receipt.get("schema_version") != "PETCT-NNUNET-ENV-v1.1":
    raise SystemExit("core environment receipt is not v1.1")
if Path(receipt.get("conda_prefix", "")).resolve() != prefix:
    raise SystemExit("core environment receipt binds a different Conda prefix")
for key in ("cc3d_import", "cc3d_distribution_root"):
    path = Path(receipt.get(key, "")).resolve()
    if not path.is_relative_to(prefix):
        raise SystemExit(f"{key} is outside the isolated core environment")
preflight = receipt.get("official_autopetv_preflight", {})
for key, path, callable_name in (
    ("simulator", simulator, "simulate_scribble_from_label"),
    ("metrics", metrics, "MetricEvaluator"),
):
    record = preflight.get(key, {})
    if (record.get("import_status") != "PASS"
            or Path(record.get("path", "")).resolve() != path
            or record.get("sha256") != sha(path)
            or record.get("required_callable") != callable_name):
        raise SystemExit(f"official AutoPET V {key} preflight binding mismatch")
print(json.dumps({"status": "PASS", "schema_version": receipt["schema_version"], "cc3d": "BOUND", "official_simulator": "BOUND", "official_metrics": "BOUND"}, sort_keys=True))
' "${CORE_ENV_RECEIPT}" "${CONDA_ENV}" "${OFFICIAL_SIMULATOR}" "${OFFICIAL_METRICS}"

readarray -t P2T_SEEDS < <("${PYTHON}" -c \
  'import json,sys; print(*json.load(open(sys.argv[1]))["p2t"]["training"]["seeds"], sep="\n")' \
  "${EXPERIMENT_CONFIG}")
readarray -t P2T_ARMS < <("${PYTHON}" -c \
  'import json,sys; print(*json.load(open(sys.argv[1], encoding="utf-8"))["p2t"]["simple_first_input_arms"], sep="\n")' \
  "${EXPERIMENT_CONFIG}")
readarray -t EDITOR_SEEDS < <("${PYTHON}" -c \
  'import json,sys; print(*json.load(open(sys.argv[1]))["editor"]["training"]["seeds"], sep="\n")' \
  "${EXPERIMENT_CONFIG}")
readarray -t EDITOR_CONDITIONS < <("${PYTHON}" -c \
  'import json,sys; print(*json.load(open(sys.argv[1]))["editor"]["conditions"], sep="\n")' \
  "${EXPERIMENT_CONFIG}")
readarray -t TRAINABLE_EDITOR_CONDITIONS < <("${PYTHON}" -c \
  'import json,sys; print(*json.load(open(sys.argv[1], encoding="utf-8"))["editor"]["training_conditions"], sep="\n")' \
  "${EXPERIMENT_CONFIG}")
P2T_PRIMARY_ARCHITECTURE="$("${PYTHON}" -c \
  'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["p2t"]["primary_architecture_id"])' \
  "${EXPERIMENT_CONFIG}")"
EDITOR_PRIMARY_ARCHITECTURE="$("${PYTHON}" -c \
  'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["editor"]["primary_architecture_id"])' \
  "${EXPERIMENT_CONFIG}")"
CONFIRMATORY_FROZEN="$("${PYTHON}" -c '
import json, sys
c = json.load(open(sys.argv[1], encoding="utf-8"))
s = c["statistics"]
thresholds = s.get("effect_thresholds")
p2t = c["p2t"].get("confirmatory_contrast")
editor = s.get("confirmatory_contrasts")
fields = set(s.get("required_frozen_contrast_fields", ()))
required = {"family", "treatment", "comparator", "metric", "threshold_ref", "null_margin", "alternative"}
def finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value == value and abs(value) != float("inf")
def valid_contrast(value):
    return isinstance(value, dict) and required <= set(value) and finite_number(value.get("null_margin")) and value.get("threshold_ref") in thresholds
ok = (
    c["p2t"].get("confirmatory_execution_gate") == "ACTIVE_AFTER_ERROR_ATLAS_FEASIBILITY_AND_EFFECT_FREEZE"
    and s.get("confirmatory_execution_gate") == "ACTIVE_AFTER_ERROR_ATLAS_FEASIBILITY_AND_EFFECT_FREEZE"
    and fields == required
    and isinstance(thresholds, dict) and bool(thresholds)
    and all(finite_number(value) for value in thresholds.values())
    and valid_contrast(p2t)
    and isinstance(editor, list) and bool(editor) and all(valid_contrast(item) for item in editor)
)
print(1 if ok else 0)
' "${EXPERIMENT_CONFIG}")"
if [[ "${EVALUATION_PARTITION}" == "test" && "${CONFIRMATORY_FROZEN}" != "1" ]]; then
  echo "test execution is blocked until v2 effect thresholds and contrast objects are frozen" >&2
  exit 37
fi
EDITOR_CONFIRMATORY_CONDITIONS=()
if [[ "${CONFIRMATORY_FROZEN}" == "1" ]]; then
  readarray -t EDITOR_CONFIRMATORY_CONDITIONS < <("${PYTHON}" -c '
import json, sys
c = json.load(open(sys.argv[1], encoding="utf-8"))
conditions = sorted({item[key] for item in c["statistics"]["confirmatory_contrasts"] for key in ("treatment", "comparator")})
print(*conditions, sep="\n")
' "${EXPERIMENT_CONFIG}")
fi

frozen_checkpoint_field() {
  local role="$1" field="$2"
  "${PYTHON}" "${PROJECT_ROOT}/scripts/common/petct_development_freeze.py" \
    resolve-checkpoint --bindings "${FROZEN_CHECKPOINT_BINDINGS}" \
    --role "${role}" --field "${field}"
}

p2t_checkpoint_role() {
  printf 'selected_checkpoint:p2t:%s:%s:seed%s\n' "$1" "$2" "$3"
}
p2t_checkpoint_path() {
  local architecture="$1" arm="$2" seed="$3" role
  if [[ "${architecture}" != "${P2T_PRIMARY_ARCHITECTURE}" ]]; then
    echo "deferred P2T architectures are not executable in the current campaign" >&2
    return 64
  fi
  if [[ "${EVALUATION_PARTITION}" == "test" ]]; then
    role="$(p2t_checkpoint_role "${architecture}" "${arm}" "${seed}")"
    frozen_checkpoint_field "${role}" path
  else
    printf '%s\n' "${RUN_ROOT}/models/p2t/${arm}_seed${seed}.pth"
  fi
}
p2t_training_manifest_path() {
  local architecture="$1" arm="$2" seed="$3" role
  if [[ "${EVALUATION_PARTITION}" == "test" ]]; then
    role="$(p2t_checkpoint_role "${architecture}" "${arm}" "${seed}")"
    frozen_checkpoint_field "${role}" training_manifest.path
  else
    printf '%s\n' "${CONTROLLED_TENSORS}"
  fi
}
editor_checkpoint_role() {
  printf 'selected_checkpoint:editor:%s:%s:seed%s\n' "$1" "${EDITOR_PRIMARY_ARCHITECTURE}" "$2"
}
editor_checkpoint_path() {
  local condition="$1" seed="$2" role
  if [[ "${EVALUATION_PARTITION}" == "test" ]]; then
    role="$(editor_checkpoint_role "${condition}" "${seed}")"
    frozen_checkpoint_field "${role}" path
  else
    printf '%s\n' "${RUN_ROOT}/models/editor/${condition}_seed${seed}.pth"
  fi
}
editor_training_manifest_path() {
  local condition="$1" seed="$2" role
  if [[ "${EVALUATION_PARTITION}" == "test" ]]; then
    role="$(editor_checkpoint_role "${condition}" "${seed}")"
    frozen_checkpoint_field "${role}" training_manifest.path
  else
    printf '%s\n' "${NATURAL_TENSORS}"
  fi
}

if [[ "${EVALUATION_PARTITION}" == "val" && ! -e "${OOF_READY}" ]]; then
  run_stage oof_5fold_parallel \
    "${PROJECT_ROOT}/scripts/baseline/run_petct_m0_oof_parallel.sh" "${GPU0_ID}" "${GPU1_ID}"
fi
run_stage validate_oof_ready \
  "${PYTHON}" "${PROJECT_ROOT}/scripts/baseline/validate_petct_m0_oof.py" \
  validate-ready-receipt "${OOF_READY}"

if [[ "${EVALUATION_PARTITION}" == "test" ]]; then
  SPLITS_FINAL="$("${PYTHON}" -c '
import hashlib, json, sys
ready = json.load(open(sys.argv[1], encoding="utf-8"))
record = ready.get("validated_bundle", {}).get("splits_final", {})
path = record.get("path", "")
with open(path, "rb") as stream:
    observed = hashlib.sha256(stream.read()).hexdigest()
if observed != record.get("sha256"):
    raise SystemExit("freeze-bound OOF_READY splits_final hash mismatch")
print(path)
' "${OOF_READY}")"
else
  SPLITS_FINAL="$("${PYTHON}" -c \
    'import json,sys; r=json.load(open(sys.argv[1])); s=json.load(open(r["campaign_spec"]["path"])); print(s["prerequisite_paths"]["splits_final"])' \
    "${FULL_TRAIN_READY}")"
fi
if [[ ! -f "${SPLITS_FINAL}" || -L "${SPLITS_FINAL}" ]]; then
  echo "FULL_TRAIN_READY points to a missing splits_final.json." >&2
  exit 5
fi

IDENTITY_CASE_MANIFEST="${RUN_ROOT}/artifacts/source_case_identity.jsonl"
run_stage source_identity_manifest \
  "${PYTHON}" "${PROJECT_ROOT}/scripts/data/build_petct_source_case_manifest.py" \
  --mode identity \
  --dataset-root "${SOURCE_DATASET}" --splits-final "${SPLITS_FINAL}" \
  --audit-json "${AUDIT_JSON}" --output "${IDENTITY_CASE_MANIFEST}"

if [[ "${EVALUATION_PARTITION}" == "val" ]]; then
  LEARNING_SPLIT="${RUN_ROOT}/artifacts/learning_split.json"
  run_stage learning_split \
    "${PYTHON}" "${PROJECT_ROOT}/scripts/data/build_petct_learning_split.py" \
    --case-manifest "${IDENTITY_CASE_MANIFEST}" --experiment-config "${EXPERIMENT_CONFIG}" \
    --output "${LEARNING_SPLIT}"
fi
run_stage validate_learning_split \
  "${PYTHON}" "${PROJECT_ROOT}/scripts/data/validate_petct_learning_split.py" \
  --split "${LEARNING_SPLIT}" --case-manifest "${IDENTITY_CASE_MANIFEST}" \
  --experiment-config "${EXPERIMENT_CONFIG}"

declare -a SELECTED_PARTITIONS=(train val)
if [[ "${EVALUATION_PARTITION}" == "test" ]]; then
  SELECTED_PARTITIONS+=(test)
fi

# The identity manifest freezes the complete 597-case membership without
# opening any image leaf.  Only after the patient split is fixed do we hash and
# header-check the explicitly authorized partitions.  Locked test rows remain
# present as identity-only records during validation runs.
CASE_MANIFEST="${RUN_ROOT}/artifacts/source_cases.jsonl"
run_stage source_manifest \
  "${PYTHON}" "${PROJECT_ROOT}/scripts/data/build_petct_source_case_manifest.py" \
  --mode materialize --identity-manifest "${IDENTITY_CASE_MANIFEST}" \
  --learning-split "${LEARNING_SPLIT}" --experiment-config "${EXPERIMENT_CONFIG}" \
  --partitions "${SELECTED_PARTITIONS[@]}" "${TEST_ACCESS_ARGS[@]}" \
  --output "${CASE_MANIFEST}"

M0_ROWS="${RUN_ROOT}/metrics/m0_oof_rows.jsonl"
M0_SUMMARY="${RUN_ROOT}/metrics/m0_oof_summary.json"
run_stage m0_metrics \
  "${PYTHON}" "${PROJECT_ROOT}/scripts/evaluation/evaluate_petct_m0_oof.py" \
  --oof-ready "${OOF_READY}" --case-manifest "${CASE_MANIFEST}" \
  --learning-split "${LEARNING_SPLIT}" --experiment-config "${EXPERIMENT_CONFIG}" \
  --official-metrics "${OFFICIAL_METRICS}" --partitions "${SELECTED_PARTITIONS[@]}" \
  --rows "${M0_ROWS}" --summary "${M0_SUMMARY}" "${TEST_ACCESS_ARGS[@]}"

PIPELINE_M0_INPUTS="${RUN_ROOT}/artifacts/pipeline_m0_evaluation_inputs.json"
run_stage pipeline_m0_evaluation_inputs "${PYTHON}" -c '
import json, os, sys
output = sys.argv[1]
keys = ("experiment_config", "case_manifest", "learning_split", "oof_ready", "official_metrics", "m0_rows", "m0_summary")
partitions_marker = sys.argv.index("--PARTITIONS")
common_marker = sys.argv.index("--COMMON")
payload = dict(zip(keys, sys.argv[2:partitions_marker]))
payload["evaluation_partitions"] = sys.argv[partitions_marker + 1:common_marker]
receipt, run_root, evaluation_partition, frozen_m0, frozen_environment, frozen_oof, f0_path, f0_sha256, f0_bytes = sys.argv[common_marker + 1:common_marker + 10]
payload["test_access_receipt"] = receipt or None
payload["run_root"] = run_root
payload["evaluation_partition"] = evaluation_partition
payload["frozen_m0_validation_receipt"] = frozen_m0 or None
payload["frozen_environment_receipt"] = frozen_environment or None
payload["frozen_oof_receipt"] = frozen_oof or None
payload["f0_readiness"] = {"path": f0_path, "sha256": f0_sha256, "bytes": int(f0_bytes)}
fd = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o640)
with os.fdopen(fd, "w", encoding="utf-8") as stream:
    json.dump(payload, stream, indent=2, sort_keys=True); stream.write("\n")
' "${PIPELINE_M0_INPUTS}" "${EXPERIMENT_CONFIG}" "${CASE_MANIFEST}" "${LEARNING_SPLIT}" \
  "${OOF_READY}" "${OFFICIAL_METRICS}" "${M0_ROWS}" "${M0_SUMMARY}" \
  --PARTITIONS "${SELECTED_PARTITIONS[@]}" --COMMON \
  "${TEST_ACCESS_RECEIPT}" "${RUN_ROOT}" "${EVALUATION_PARTITION}" \
  "${FROZEN_M0_VALIDATION_RECEIPT}" "${FROZEN_ENVIRONMENT_RECEIPT}" \
  "${FROZEN_OOF_RECEIPT}" "${F0_READY_PATH}" "${F0_READY_SHA256}" \
  "${F0_READY_BYTES}"
run_stage validate_m0_evaluation \
  "${PYTHON}" "${PROJECT_ROOT}/scripts/orchestration/validate_petct_route_a_receipt_pipeline.py" \
  --inputs "${PIPELINE_M0_INPUTS}" --target m0_evaluation \
  --output "${RUN_ROOT}/artifacts/M0_EVALUATION_READY.json"

RESIDUAL_MANIFEST="${RUN_ROOT}/artifacts/residuals.jsonl"
RESIDUAL_READY="${RUN_ROOT}/artifacts/RESIDUAL_READY.json"
run_stage residual_manifest \
  "${PYTHON}" "${PROJECT_ROOT}/scripts/data/build_petct_residual_manifest.py" \
  --oof-ready "${OOF_READY}" --learning-split "${LEARNING_SPLIT}" \
  --experiment-config "${EXPERIMENT_CONFIG}" --case-manifest "${CASE_MANIFEST}" \
  --partitions "${SELECTED_PARTITIONS[@]}" "${TEST_ACCESS_ARGS[@]}" \
  --output-dir "${RUN_ROOT}/artifacts/residual_masks" \
  --output-manifest "${RESIDUAL_MANIFEST}" \
  --ready-receipt "${RESIDUAL_READY}"

# Freeze the professor-directed six-class ontology after OOF/error-atlas
# construction and before either controlled or natural cue generation.
INTENT_TAXONOMY_FREEZE="${RUN_ROOT}/artifacts/intent_taxonomy_freeze.json"
run_stage freeze_intent_taxonomy \
  "${PYTHON}" "${PROJECT_ROOT}/scripts/data/freeze_petct_intent_taxonomy.py" \
  --experiment-config "${EXPERIMENT_CONFIG}" \
  --output "${INTENT_TAXONOMY_FREEZE}"

CONTROLLED_EPISODES="${RUN_ROOT}/artifacts/controlled_episodes.jsonl"
CONTROLLED_DATA_READY="${RUN_ROOT}/artifacts/CONTROLLED_DATA_READY.json"
CONTROLLED_BUILD=(
  "${PYTHON}" "${PROJECT_ROOT}/scripts/p2t/build_petct_matched_state_dataset.py"
  --case-manifest "${CASE_MANIFEST}" --learning-split "${LEARNING_SPLIT}"
  --experiment-config "${EXPERIMENT_CONFIG}" --official-simulator "${OFFICIAL_SIMULATOR}"
  --official-runtime-manifest "${OFFICIAL_RUNTIME_MANIFEST}"
  --partitions "${SELECTED_PARTITIONS[@]}"
  --state-root "${RUN_ROOT}/artifacts/controlled_states"
  --visible-root "${RUN_ROOT}/artifacts/controlled_episode_visible"
  --evaluation-root "${RUN_ROOT}/artifacts/controlled_episode_evaluation"
  --output-manifest "${CONTROLLED_EPISODES}"
  --exclusions "${RUN_ROOT}/artifacts/controlled_exclusions.jsonl"
  --ready-receipt "${CONTROLLED_DATA_READY}"
  "${TEST_ACCESS_ARGS[@]}"
)
run_stage controlled_matched_states "${CONTROLLED_BUILD[@]}"

# The residual builder already applies the explicit authorized partition set
# and publishes a hash-bound leaf receipt.  Use that artifact directly; an
# unreceipted filtered copy would sever natural scribbles from its denominator.
SCOPED_RESIDUALS="${RESIDUAL_MANIFEST}"

NATURAL_EPISODES="${RUN_ROOT}/artifacts/natural_episodes.jsonl"
NATURAL_PRIMARY_DATA_READY="${RUN_ROOT}/artifacts/NATURAL_PRIMARY_DATA_READY.json"
run_stage natural_autopetv_scribbles \
  "${PYTHON}" "${PROJECT_ROOT}/scripts/data/build_petct_scribble_dataset.py" \
  --residual-manifest "${SCOPED_RESIDUALS}" --residual-ready "${RESIDUAL_READY}" \
  --official-simulator "${OFFICIAL_SIMULATOR}" \
  --official-runtime-manifest "${OFFICIAL_RUNTIME_MANIFEST}" \
  --experiment-config "${EXPERIMENT_CONFIG}" --lane natural --oof-ready "${OOF_READY}" \
  --learning-split "${LEARNING_SPLIT}" --partitions "${SELECTED_PARTITIONS[@]}" \
  "${TEST_ACCESS_ARGS[@]}" \
  --visible-root "${RUN_ROOT}/artifacts/natural_episode_visible" \
  --evaluation-root "${RUN_ROOT}/artifacts/natural_episode_evaluation" \
  --authorized-root "${RUN_ROOT}/artifacts/natural_authorized" \
  --output-manifest "${NATURAL_EPISODES}" \
  --exclusions "${RUN_ROOT}/artifacts/natural_exclusions.jsonl" \
  --ready-receipt "${NATURAL_PRIMARY_DATA_READY}"

CONTROLLED_TENSORS="${RUN_ROOT}/artifacts/controlled_tensors.jsonl"
NATURAL_TENSORS="${RUN_ROOT}/artifacts/natural_tensors.jsonl"
run_stage controlled_tensors \
  "${PYTHON}" "${PROJECT_ROOT}/scripts/data/materialize_petct_learning_tensors.py" \
  --episode-manifest "${CONTROLLED_EPISODES}" \
  --visible-root "${RUN_ROOT}/artifacts/controlled_tensor_visible" \
  --evaluation-root "${RUN_ROOT}/artifacts/controlled_tensor_evaluation" \
  --output-manifest "${CONTROLLED_TENSORS}" --experiment-config "${EXPERIMENT_CONFIG}" \
  --learning-split "${LEARNING_SPLIT}" --partitions "${SELECTED_PARTITIONS[@]}" \
  "${TEST_ACCESS_ARGS[@]}"
run_stage natural_tensors \
  "${PYTHON}" "${PROJECT_ROOT}/scripts/data/materialize_petct_learning_tensors.py" \
  --episode-manifest "${NATURAL_EPISODES}" \
  --visible-root "${RUN_ROOT}/artifacts/natural_tensor_visible" \
  --evaluation-root "${RUN_ROOT}/artifacts/natural_tensor_evaluation" \
  --output-manifest "${NATURAL_TENSORS}" --experiment-config "${EXPERIMENT_CONFIG}" \
  --learning-split "${LEARNING_SPLIT}" --partitions "${SELECTED_PARTITIONS[@]}" \
  "${TEST_ACCESS_ARGS[@]}"

PIPELINE_DATA_INPUTS="${RUN_ROOT}/artifacts/pipeline_data_inputs.json"
run_stage pipeline_data_inputs "${PYTHON}" -c '
import json, os, sys
keys = ("experiment_config", "case_manifest", "learning_split", "controlled_episode_manifest", "controlled_tensor_manifest", "oof_ready", "natural_episode_manifest", "natural_tensor_manifest", "m0_evaluation_ready", "intent_taxonomy_freeze")
payload = dict(zip(keys, sys.argv[2:12]))
payload["test_access_receipt"] = sys.argv[12] or None
payload["run_root"] = sys.argv[13]
payload["evaluation_partition"] = sys.argv[14]
payload["frozen_checkpoint_bindings"] = sys.argv[15] or None
payload["frozen_m0_validation_receipt"] = sys.argv[16] or None
payload["frozen_environment_receipt"] = sys.argv[17] or None
payload["frozen_oof_receipt"] = sys.argv[18] or None
payload["f0_readiness"] = {"path": sys.argv[19], "sha256": sys.argv[20], "bytes": int(sys.argv[21])}
payload["residual_ready"] = sys.argv[22]
payload["controlled_data_ready"] = sys.argv[23]
payload["natural_primary_data_ready"] = sys.argv[24]
fd = os.open(sys.argv[1], os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o640)
with os.fdopen(fd, "w", encoding="utf-8") as stream:
    json.dump(payload, stream, indent=2, sort_keys=True); stream.write("\n")
' "${PIPELINE_DATA_INPUTS}" "${EXPERIMENT_CONFIG}" "${CASE_MANIFEST}" "${LEARNING_SPLIT}" \
  "${CONTROLLED_EPISODES}" "${CONTROLLED_TENSORS}" "${OOF_READY}" \
  "${NATURAL_EPISODES}" "${NATURAL_TENSORS}" \
  "${RUN_ROOT}/artifacts/M0_EVALUATION_READY.json" "${INTENT_TAXONOMY_FREEZE}" \
  "${TEST_ACCESS_RECEIPT}" "${RUN_ROOT}" "${EVALUATION_PARTITION}" \
  "${FROZEN_CHECKPOINT_BINDINGS}" \
  "${FROZEN_M0_VALIDATION_RECEIPT}" "${FROZEN_ENVIRONMENT_RECEIPT}" \
  "${FROZEN_OOF_RECEIPT}" "${F0_READY_PATH}" "${F0_READY_SHA256}" \
  "${F0_READY_BYTES}" "${RESIDUAL_READY}" "${CONTROLLED_DATA_READY}" \
  "${NATURAL_PRIMARY_DATA_READY}"
run_stage validate_p2t_data \
  "${PYTHON}" "${PROJECT_ROOT}/scripts/orchestration/validate_petct_route_a_receipt_pipeline.py" \
  --inputs "${PIPELINE_DATA_INPUTS}" --target p2t_data \
  --output "${RUN_ROOT}/artifacts/P2T_DATA_READY.json"
run_stage validate_editor_data \
  "${PYTHON}" "${PROJECT_ROOT}/scripts/orchestration/validate_petct_route_a_receipt_pipeline.py" \
  --inputs "${PIPELINE_DATA_INPUTS}" --target editor_data \
  --output "${RUN_ROOT}/artifacts/EDITOR_DATA_READY.json"

declare -a P2T_JOBS=()
for seed in "${P2T_SEEDS[@]}"; do
  for arm in "${P2T_ARMS[@]}"; do
    P2T_JOBS+=("${seed}|${arm}")
  done
done
run_p2t_train_queue() {
  local gpu="$1" parity="$2" index job seed arm checkpoint
  for index in "${!P2T_JOBS[@]}"; do
    (( index % 2 == parity )) || continue
    job="${P2T_JOBS[index]}"; seed="${job%%|*}"; arm="${job#*|}"
    checkpoint="${RUN_ROOT}/models/p2t/${arm}_seed${seed}.pth"
    CUDA_VISIBLE_DEVICES="${gpu}" run_stage "p2t_train_${arm}_seed${seed}" \
      "${PYTHON}" "${PROJECT_ROOT}/scripts/p2t/train_petct_p2t.py" \
      --manifest "${CONTROLLED_TENSORS}" --experiment-config "${EXPERIMENT_CONFIG}" \
      --learning-split "${LEARNING_SPLIT}" \
      --output "${checkpoint}" --seed "${seed}" --input-ablation "${arm}" --device cuda:0
  done
}
if [[ "${EVALUATION_PARTITION}" == "val" ]]; then
  run_p2t_train_queue "${GPU0_ID}" 0 & P2T_TRAIN_GPU0_PID=$!
  run_p2t_train_queue "${GPU1_ID}" 1 & P2T_TRAIN_GPU1_PID=$!
  set +e
  wait "${P2T_TRAIN_GPU0_PID}"; P2T_TRAIN_GPU0_STATUS=$?
  wait "${P2T_TRAIN_GPU1_PID}"; P2T_TRAIN_GPU1_STATUS=$?
  set -e
  if [[ ${P2T_TRAIN_GPU0_STATUS} -ne 0 || ${P2T_TRAIN_GPU1_STATUS} -ne 0 ]]; then
    echo "P2T training queues failed: gpu0=${P2T_TRAIN_GPU0_STATUS} gpu1=${P2T_TRAIN_GPU1_STATUS}" >&2
    exit 30
  fi
fi

run_p2t_eval_queue() {
  local gpu="$1" parity="$2" index job seed arm checkpoint training_manifest predictions paired metrics
  for index in "${!P2T_JOBS[@]}"; do
    (( index % 2 == parity )) || continue
    job="${P2T_JOBS[index]}"; seed="${job%%|*}"; arm="${job#*|}"
    checkpoint="$(p2t_checkpoint_path "${P2T_PRIMARY_ARCHITECTURE}" "${arm}" "${seed}")"
    training_manifest="$(p2t_training_manifest_path "${P2T_PRIMARY_ARCHITECTURE}" "${arm}" "${seed}")"
    predictions="${RUN_ROOT}/metrics/p2t_${arm}_seed${seed}_${EVALUATION_PARTITION}_predictions.jsonl"
    paired="${RUN_ROOT}/metrics/p2t_${arm}_seed${seed}_${EVALUATION_PARTITION}_paired.jsonl"
    metrics="${RUN_ROOT}/metrics/p2t_${arm}_seed${seed}_${EVALUATION_PARTITION}.json"
    CUDA_VISIBLE_DEVICES="${gpu}" run_stage "p2t_eval_${arm}_seed${seed}" \
      "${PYTHON}" "${PROJECT_ROOT}/scripts/evaluation/evaluate_petct_p2t.py" \
      --manifest "${CONTROLLED_TENSORS}" --training-manifest "${training_manifest}" \
      --experiment-config "${EXPERIMENT_CONFIG}" \
      --learning-split "${LEARNING_SPLIT}" \
      --checkpoint "${checkpoint}" --partition "${EVALUATION_PARTITION}" \
      --predictions "${predictions}" --paired-evaluation-rows "${paired}" \
      --metrics "${metrics}" --device cuda:0 "${TEST_ACCESS_ARGS[@]}"
  done
}
run_p2t_eval_queue "${GPU0_ID}" 0 & P2T_GPU0_PID=$!
run_p2t_eval_queue "${GPU1_ID}" 1 & P2T_GPU1_PID=$!
set +e
wait "${P2T_GPU0_PID}"; P2T_GPU0_STATUS=$?
wait "${P2T_GPU1_PID}"; P2T_GPU1_STATUS=$?
set -e
if [[ ${P2T_GPU0_STATUS} -ne 0 || ${P2T_GPU1_STATUS} -ne 0 ]]; then
  echo "P2T evaluation queues failed: gpu0=${P2T_GPU0_STATUS} gpu1=${P2T_GPU1_STATUS}" >&2
  exit 30
fi

declare -a P2T_METRICS=() P2T_CHECKPOINTS=() P2T_PREDICTIONS=() P2T_PAIRED_ROWS=()
declare -a P2T_CONFIRMATORY_ARGS=()
for seed in "${P2T_SEEDS[@]}"; do
  for arm in "${P2T_ARMS[@]}"; do
    checkpoint="$(p2t_checkpoint_path "${P2T_PRIMARY_ARCHITECTURE}" "${arm}" "${seed}")"
    P2T_CHECKPOINTS+=("${checkpoint}")
    P2T_PREDICTIONS+=("${RUN_ROOT}/metrics/p2t_${arm}_seed${seed}_${EVALUATION_PARTITION}_predictions.jsonl")
    P2T_PAIRED_ROWS+=("${RUN_ROOT}/metrics/p2t_${arm}_seed${seed}_${EVALUATION_PARTITION}_paired.jsonl")
    P2T_METRICS+=("${RUN_ROOT}/metrics/p2t_${arm}_seed${seed}_${EVALUATION_PARTITION}.json")
    if [[ "${arm}" == "full" || "${arm}" == "no_M0" ]]; then
      P2T_CONFIRMATORY_ARGS+=(
        --run "${seed}" "${arm}"
        "${checkpoint}"
        "${RUN_ROOT}/metrics/p2t_${arm}_seed${seed}_${EVALUATION_PARTITION}.json"
        "${RUN_ROOT}/metrics/p2t_${arm}_seed${seed}_${EVALUATION_PARTITION}_paired.jsonl"
      )
    fi
  done
done
P2T_CONFIRMATORY="${RUN_ROOT}/metrics/p2t_confirmatory_${EVALUATION_PARTITION}.json"
if [[ "${CONFIRMATORY_FROZEN}" == "1" ]]; then
  run_stage p2t_confirmatory \
    "${PYTHON}" "${PROJECT_ROOT}/scripts/evaluation/aggregate_petct_p2t_confirmatory.py" \
    "${P2T_CONFIRMATORY_ARGS[@]}" --experiment-config "${EXPERIMENT_CONFIG}" \
    --output "${P2T_CONFIRMATORY}"
else
  run_stage p2t_descriptive "${PYTHON}" -c '
import hashlib, json, os, sys
output, config, partition, *paths = sys.argv[1:]
def record(path):
    data = open(path, "rb").read()
    return {"path": os.path.abspath(path), "sha256": hashlib.sha256(data).hexdigest()}
payload = {
    "schema_version": "PETCT-P2T-DESCRIPTIVE-v2.0",
    "analysis_status": "DESCRIPTIVE_ONLY_PENDING_EFFECT_FREEZE",
    "partition": partition,
    "experiment_config_sha256": hashlib.sha256(open(config, "rb").read()).hexdigest(),
    "input_runs": [record(path) for path in paths],
    "hypothesis_verdict": None,
    "confirmatory_eligible": False,
}
fd = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o640)
with os.fdopen(fd, "w", encoding="utf-8") as stream:
    json.dump(payload, stream, indent=2, sort_keys=True); stream.write("\n")
' "${P2T_CONFIRMATORY}" "${EXPERIMENT_CONFIG}" "${EVALUATION_PARTITION}" \
    "${P2T_METRICS[@]}" "${P2T_PREDICTIONS[@]}" "${P2T_PAIRED_ROWS[@]}"
fi

PIPELINE_P2T_INPUTS="${RUN_ROOT}/artifacts/pipeline_p2t_inputs.json"
run_stage pipeline_p2t_inputs "${PYTHON}" -c '
import json, os, sys
output, base, partition, confirmatory, p2t_data_ready = sys.argv[1:6]
payload = json.load(open(base, encoding="utf-8"))
payload["evaluation_partition"] = partition
payload["p2t_confirmatory"] = confirmatory
payload["p2t_data_ready"] = p2t_data_ready
markers = {name: sys.argv.index(name) for name in ("--METRICS", "--CHECKPOINTS", "--PREDICTIONS", "--PAIRED")}
payload["p2t_metrics"] = sys.argv[markers["--METRICS"] + 1:markers["--CHECKPOINTS"]]
payload["p2t_checkpoints"] = sys.argv[markers["--CHECKPOINTS"] + 1:markers["--PREDICTIONS"]]
payload["p2t_predictions"] = sys.argv[markers["--PREDICTIONS"] + 1:markers["--PAIRED"]]
payload["p2t_paired_rows"] = sys.argv[markers["--PAIRED"] + 1:]
fd = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o640)
with os.fdopen(fd, "w", encoding="utf-8") as stream:
    json.dump(payload, stream, indent=2, sort_keys=True); stream.write("\n")
' "${PIPELINE_P2T_INPUTS}" "${PIPELINE_DATA_INPUTS}" "${EVALUATION_PARTITION}" \
  "${P2T_CONFIRMATORY}" "${RUN_ROOT}/artifacts/P2T_DATA_READY.json" \
  --METRICS "${P2T_METRICS[@]}" --CHECKPOINTS "${P2T_CHECKPOINTS[@]}" \
  --PREDICTIONS "${P2T_PREDICTIONS[@]}" --PAIRED "${P2T_PAIRED_ROWS[@]}"
run_stage validate_p2t_results \
  "${PYTHON}" "${PROJECT_ROOT}/scripts/orchestration/validate_petct_route_a_receipt_pipeline.py" \
  --inputs "${PIPELINE_P2T_INPUTS}" --target p2t_results \
  --output "${RUN_ROOT}/artifacts/P2T_RESULTS_READY.json"

# Apply only the frozen primary full-input P2T arm to the natural OOF tensor
# manifest.  The CLI verifies the controlled training-manifest hash against
# the checkpoint and separately records the natural inference-manifest hash.
run_p2t_natural_queue() {
  local gpu="$1" parity="$2" index seed checkpoint training_manifest predictions paired metrics
  for index in "${!P2T_SEEDS[@]}"; do
    (( index % 2 == parity )) || continue
    seed="${P2T_SEEDS[index]}"
    checkpoint="$(p2t_checkpoint_path "${P2T_PRIMARY_ARCHITECTURE}" full "${seed}")"
    training_manifest="$(p2t_training_manifest_path "${P2T_PRIMARY_ARCHITECTURE}" full "${seed}")"
    predictions="${RUN_ROOT}/metrics/p2t_natural/full_seed${seed}_${EVALUATION_PARTITION}_predictions.jsonl"
    paired="${RUN_ROOT}/metrics/p2t_natural/full_seed${seed}_${EVALUATION_PARTITION}_paired.jsonl"
    metrics="${RUN_ROOT}/metrics/p2t_natural/full_seed${seed}_${EVALUATION_PARTITION}.json"
    CUDA_VISIBLE_DEVICES="${gpu}" run_stage "p2t_natural_infer_full_seed${seed}" \
      "${PYTHON}" "${PROJECT_ROOT}/scripts/p2t/infer_petct_p2t.py" \
      --manifest "${NATURAL_TENSORS}" --training-manifest "${training_manifest}" \
      --experiment-config "${EXPERIMENT_CONFIG}" --learning-split "${LEARNING_SPLIT}" \
      --checkpoint "${checkpoint}" \
      --partition "${EVALUATION_PARTITION}" --predictions "${predictions}" \
      --paired-evaluation-rows "${paired}" --metrics "${metrics}" --device cuda:0 \
      "${TEST_ACCESS_ARGS[@]}"
  done
}
run_p2t_natural_queue "${GPU0_ID}" 0 & P2T_NATURAL_GPU0_PID=$!
run_p2t_natural_queue "${GPU1_ID}" 1 & P2T_NATURAL_GPU1_PID=$!
set +e
wait "${P2T_NATURAL_GPU0_PID}"; P2T_NATURAL_GPU0_STATUS=$?
wait "${P2T_NATURAL_GPU1_PID}"; P2T_NATURAL_GPU1_STATUS=$?
set -e
if [[ ${P2T_NATURAL_GPU0_STATUS} -ne 0 || ${P2T_NATURAL_GPU1_STATUS} -ne 0 ]]; then
  echo "Natural-lane P2T inference failed: gpu0=${P2T_NATURAL_GPU0_STATUS} gpu1=${P2T_NATURAL_GPU1_STATUS}" >&2
  exit 33
fi

declare -a P2T_NATURAL_METRICS=() P2T_NATURAL_PREDICTIONS=() P2T_NATURAL_PAIRED_ROWS=()
for seed in "${P2T_SEEDS[@]}"; do
  P2T_NATURAL_METRICS+=("${RUN_ROOT}/metrics/p2t_natural/full_seed${seed}_${EVALUATION_PARTITION}.json")
  P2T_NATURAL_PREDICTIONS+=("${RUN_ROOT}/metrics/p2t_natural/full_seed${seed}_${EVALUATION_PARTITION}_predictions.jsonl")
  P2T_NATURAL_PAIRED_ROWS+=("${RUN_ROOT}/metrics/p2t_natural/full_seed${seed}_${EVALUATION_PARTITION}_paired.jsonl")
done

declare -a EDITOR_TRAIN_JOBS=()
for seed in "${EDITOR_SEEDS[@]}"; do
  for condition in "${TRAINABLE_EDITOR_CONDITIONS[@]}"; do
    EDITOR_TRAIN_JOBS+=("${seed}|${condition}")
  done
done
run_editor_train_queue() {
  local gpu="$1" parity="$2" index job seed condition
  for index in "${!EDITOR_TRAIN_JOBS[@]}"; do
    (( index % 2 == parity )) || continue
    job="${EDITOR_TRAIN_JOBS[index]}"; seed="${job%%|*}"; condition="${job#*|}"
    CUDA_VISIBLE_DEVICES="${gpu}" run_stage "editor_train_${condition}_seed${seed}" \
      "${PYTHON}" "${PROJECT_ROOT}/scripts/editor/train_petct_residual_editor.py" \
      --manifest "${NATURAL_TENSORS}" --experiment-config "${EXPERIMENT_CONFIG}" \
      --learning-split "${LEARNING_SPLIT}" \
      --condition "${condition}" \
      --output "${RUN_ROOT}/models/editor/${condition}_seed${seed}.pth" \
      --seed "${seed}" --device cuda:0
  done
}
if [[ "${EVALUATION_PARTITION}" == "val" ]]; then
  run_editor_train_queue "${GPU0_ID}" 0 & EDITOR_GPU0_PID=$!
  run_editor_train_queue "${GPU1_ID}" 1 & EDITOR_GPU1_PID=$!
  set +e
  wait "${EDITOR_GPU0_PID}"; EDITOR_GPU0_STATUS=$?
  wait "${EDITOR_GPU1_PID}"; EDITOR_GPU1_STATUS=$?
  set -e
  if [[ ${EDITOR_GPU0_STATUS} -ne 0 || ${EDITOR_GPU1_STATUS} -ne 0 ]]; then
    echo "Editor training queues failed: gpu0=${EDITOR_GPU0_STATUS} gpu1=${EDITOR_GPU1_STATUS}" >&2
    exit 31
  fi
fi

INTERVENTIONS="${RUN_ROOT}/artifacts/intent_interventions_${EVALUATION_PARTITION}.jsonl"
run_stage intent_interventions \
  "${PYTHON}" "${PROJECT_ROOT}/scripts/editor/build_petct_intent_interventions.py" \
  --learning-manifest "${NATURAL_TENSORS}" --experiment-config "${EXPERIMENT_CONFIG}" \
  --learning-split "${LEARNING_SPLIT}" --partition "${EVALUATION_PARTITION}" \
  --output "${INTERVENTIONS}" "${TEST_ACCESS_ARGS[@]}"

# The config is the single exact condition inventory.  It must contain the
# six-class operation-only replacement and must not resurrect scribble_plus_ADD.
EXECUTABLE_EDITOR_CONDITIONS=("${EDITOR_CONDITIONS[@]}")
if [[ " ${EXECUTABLE_EDITOR_CONDITIONS[*]} " == *" scribble_plus_ADD "* ]]; then
  echo "obsolete scribble_plus_ADD condition is forbidden" >&2
  exit 36
fi
editor_checkpoint_condition() {
  case "$1" in
    scribble_plus_operation|same_weight_NULL|same_weight_wrong_scope|same_weight_shuffled|wrong_operation_OOD)
      printf '%s\n' scribble_plus_intent ;;
    oracle_slots|predicted_slots) printf '%s\n' scribble_plus_intent ;;
    *) printf '%s\n' "$1" ;;
  esac
}
declare -a EDITOR_EVAL_JOBS=()
for seed in "${EDITOR_SEEDS[@]}"; do
  for condition in "${EXECUTABLE_EDITOR_CONDITIONS[@]}"; do
    EDITOR_EVAL_JOBS+=("${seed}|${condition}")
  done
done
run_editor_eval_queue() {
  local gpu="$1" parity="$2" index job seed condition trained checkpoint training_manifest out_dir manifest rows summary
  local -a extras
  for index in "${!EDITOR_EVAL_JOBS[@]}"; do
    (( index % 2 == parity )) || continue
    job="${EDITOR_EVAL_JOBS[index]}"; seed="${job%%|*}"; condition="${job#*|}"
    trained="$(editor_checkpoint_condition "${condition}")"
    checkpoint="$(editor_checkpoint_path "${trained}" "${seed}")"
    training_manifest="$(editor_training_manifest_path "${trained}" "${seed}")"
    out_dir="${RUN_ROOT}/artifacts/editor_predictions/${condition}_seed${seed}_${EVALUATION_PARTITION}"
    manifest="${RUN_ROOT}/artifacts/editor_${condition}_seed${seed}_${EVALUATION_PARTITION}.jsonl"
    rows="${RUN_ROOT}/metrics/editor_${condition}_seed${seed}_${EVALUATION_PARTITION}_rows.jsonl"
    summary="${RUN_ROOT}/metrics/editor_${condition}_seed${seed}_${EVALUATION_PARTITION}.json"
    extras=()
    if [[ "${condition}" == "same_weight_shuffled" ]]; then
      extras+=(--intervention-manifest "${INTERVENTIONS}")
    fi
    if [[ "${condition}" == "predicted_slots" ]]; then
      extras+=(
        --p2t-checkpoint "$(p2t_checkpoint_path "${P2T_PRIMARY_ARCHITECTURE}" full "${seed}")"
        --p2t-predictions "${RUN_ROOT}/metrics/p2t_natural/full_seed${seed}_${EVALUATION_PARTITION}_predictions.jsonl"
        --p2t-metrics "${RUN_ROOT}/metrics/p2t_natural/full_seed${seed}_${EVALUATION_PARTITION}.json"
      )
    fi
    CUDA_VISIBLE_DEVICES="${gpu}" run_stage "editor_infer_${condition}_seed${seed}" \
      "${PYTHON}" "${PROJECT_ROOT}/scripts/editor/infer_petct_residual_editor.py" \
      --manifest "${NATURAL_TENSORS}" --training-manifest "${training_manifest}" \
      --experiment-config "${EXPERIMENT_CONFIG}" \
      --learning-split "${LEARNING_SPLIT}" \
      --checkpoint "${checkpoint}" --partition "${EVALUATION_PARTITION}" \
      --condition "${condition}" "${extras[@]}" --output-dir "${out_dir}" \
      --output-manifest "${manifest}" --device cuda:0 "${TEST_ACCESS_ARGS[@]}"
    run_stage "editor_metrics_${condition}_seed${seed}" \
      "${PYTHON}" "${PROJECT_ROOT}/scripts/evaluation/evaluate_petct_correction.py" \
      --prediction-manifest "${manifest}" --rows "${rows}" --summary "${summary}" \
      --experiment-config "${EXPERIMENT_CONFIG}" --learning-split "${LEARNING_SPLIT}" \
      --official-metrics "${OFFICIAL_METRICS}" \
      "${TEST_ACCESS_ARGS[@]}"
  done
}
run_editor_eval_queue "${GPU0_ID}" 0 & EDITOR_EVAL_GPU0_PID=$!
run_editor_eval_queue "${GPU1_ID}" 1 & EDITOR_EVAL_GPU1_PID=$!
set +e
wait "${EDITOR_EVAL_GPU0_PID}"; EDITOR_EVAL_GPU0_STATUS=$?
wait "${EDITOR_EVAL_GPU1_PID}"; EDITOR_EVAL_GPU1_STATUS=$?
set -e
if [[ ${EDITOR_EVAL_GPU0_STATUS} -ne 0 || ${EDITOR_EVAL_GPU1_STATUS} -ne 0 ]]; then
  echo "Editor evaluation queues failed: gpu0=${EDITOR_EVAL_GPU0_STATUS} gpu1=${EDITOR_EVAL_GPU1_STATUS}" >&2
  exit 32
fi

declare -a EDITOR_SUMMARIES=() EDITOR_ROWS=() EDITOR_CHECKPOINTS=()
declare -a EDITOR_CONFIRMATORY_ARGS=()
for seed in "${EDITOR_SEEDS[@]}"; do
  for condition in "${EXECUTABLE_EDITOR_CONDITIONS[@]}"; do
    trained="$(editor_checkpoint_condition "${condition}")"
    EDITOR_SUMMARIES+=("${RUN_ROOT}/metrics/editor_${condition}_seed${seed}_${EVALUATION_PARTITION}.json")
    EDITOR_ROWS+=("${RUN_ROOT}/metrics/editor_${condition}_seed${seed}_${EVALUATION_PARTITION}_rows.jsonl")
    if [[ " ${EDITOR_CONFIRMATORY_CONDITIONS[*]} " == *" ${condition} "* ]]; then
      EDITOR_CONFIRMATORY_ARGS+=(
        --run "${seed}" "${condition}"
        "$(editor_checkpoint_path "${trained}" "${seed}")"
        "${RUN_ROOT}/metrics/editor_${condition}_seed${seed}_${EVALUATION_PARTITION}.json"
        "${RUN_ROOT}/metrics/editor_${condition}_seed${seed}_${EVALUATION_PARTITION}_rows.jsonl"
      )
    fi
  done
  for condition in "${TRAINABLE_EDITOR_CONDITIONS[@]}"; do
    EDITOR_CHECKPOINTS+=("$(editor_checkpoint_path "${condition}" "${seed}")")
  done
done
EDITOR_CONFIRMATORY="${RUN_ROOT}/metrics/editor_confirmatory_${EVALUATION_PARTITION}.json"
if [[ "${CONFIRMATORY_FROZEN}" == "1" ]]; then
  run_stage editor_confirmatory \
    "${PYTHON}" "${PROJECT_ROOT}/scripts/evaluation/aggregate_petct_condition_metrics.py" \
    "${EDITOR_CONFIRMATORY_ARGS[@]}" --experiment-config "${EXPERIMENT_CONFIG}" \
    --output "${EDITOR_CONFIRMATORY}"
else
  run_stage editor_descriptive "${PYTHON}" -c '
import hashlib, json, os, sys
output, config, partition, *paths = sys.argv[1:]
def record(path):
    data = open(path, "rb").read()
    return {"path": os.path.abspath(path), "sha256": hashlib.sha256(data).hexdigest()}
payload = {
    "schema_version": "PETCT-EDITOR-DESCRIPTIVE-v2.0",
    "analysis_status": "DESCRIPTIVE_ONLY_PENDING_EFFECT_FREEZE",
    "partition": partition,
    "experiment_config_sha256": hashlib.sha256(open(config, "rb").read()).hexdigest(),
    "input_runs": [record(path) for path in paths],
    "family_verdicts": None,
    "hypothesis_verdict": None,
    "confirmatory_eligible": False,
}
fd = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o640)
with os.fdopen(fd, "w", encoding="utf-8") as stream:
    json.dump(payload, stream, indent=2, sort_keys=True); stream.write("\n")
' "${EDITOR_CONFIRMATORY}" "${EXPERIMENT_CONFIG}" "${EVALUATION_PARTITION}" \
    "${EDITOR_SUMMARIES[@]}" "${EDITOR_ROWS[@]}"
fi

PIPELINE_EDITOR_INPUTS="${RUN_ROOT}/artifacts/pipeline_editor_inputs.json"
run_stage pipeline_editor_inputs "${PYTHON}" -c '
import json, os, sys
output, base, confirmatory, editor_data_ready, p2t_results_ready = sys.argv[1:6]
payload = json.load(open(base, encoding="utf-8"))
payload["editor_confirmatory"] = confirmatory
payload["editor_data_ready"] = editor_data_ready
payload["p2t_results_ready"] = p2t_results_ready
markers = {name: sys.argv.index(name) for name in ("--NATURAL-METRICS", "--NATURAL-PREDICTIONS", "--NATURAL-PAIRED", "--SUMMARIES", "--ROWS", "--CHECKPOINTS")}
payload["p2t_natural_metrics"] = sys.argv[markers["--NATURAL-METRICS"] + 1:markers["--NATURAL-PREDICTIONS"]]
payload["p2t_natural_predictions"] = sys.argv[markers["--NATURAL-PREDICTIONS"] + 1:markers["--NATURAL-PAIRED"]]
payload["p2t_natural_paired_rows"] = sys.argv[markers["--NATURAL-PAIRED"] + 1:markers["--SUMMARIES"]]
payload["editor_summaries"] = sys.argv[markers["--SUMMARIES"] + 1:markers["--ROWS"]]
payload["editor_rows"] = sys.argv[markers["--ROWS"] + 1:markers["--CHECKPOINTS"]]
payload["editor_checkpoints"] = sys.argv[markers["--CHECKPOINTS"] + 1:]
fd = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o640)
with os.fdopen(fd, "w", encoding="utf-8") as stream:
    json.dump(payload, stream, indent=2, sort_keys=True); stream.write("\n")
' "${PIPELINE_EDITOR_INPUTS}" "${PIPELINE_P2T_INPUTS}" "${EDITOR_CONFIRMATORY}" \
  "${RUN_ROOT}/artifacts/EDITOR_DATA_READY.json" \
  "${RUN_ROOT}/artifacts/P2T_RESULTS_READY.json" \
  --NATURAL-METRICS "${P2T_NATURAL_METRICS[@]}" \
  --NATURAL-PREDICTIONS "${P2T_NATURAL_PREDICTIONS[@]}" \
  --NATURAL-PAIRED "${P2T_NATURAL_PAIRED_ROWS[@]}" \
  --SUMMARIES "${EDITOR_SUMMARIES[@]}" --ROWS "${EDITOR_ROWS[@]}" \
  --CHECKPOINTS "${EDITOR_CHECKPOINTS[@]}"
run_stage validate_editor_results \
  "${PYTHON}" "${PROJECT_ROOT}/scripts/orchestration/validate_petct_route_a_receipt_pipeline.py" \
  --inputs "${PIPELINE_EDITOR_INPUTS}" --target editor_results \
  --output "${RUN_ROOT}/artifacts/EDITOR_RESULTS_READY.json"

PIPELINE_COMPLETE_INPUTS="${RUN_ROOT}/artifacts/pipeline_complete_inputs.json"
run_stage pipeline_complete_inputs "${PYTHON}" -c '
import json, os, sys
output, base, editor_results_ready = sys.argv[1:4]
payload = json.load(open(base, encoding="utf-8"))
payload["editor_results_ready"] = editor_results_ready
marker = sys.argv.index("--EXPECTED")
payload["expected_result_artifacts"] = sys.argv[marker + 1:]
fd = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o640)
with os.fdopen(fd, "w", encoding="utf-8") as stream:
    json.dump(payload, stream, indent=2, sort_keys=True); stream.write("\n")
' "${PIPELINE_COMPLETE_INPUTS}" "${PIPELINE_EDITOR_INPUTS}" \
  "${RUN_ROOT}/artifacts/EDITOR_RESULTS_READY.json" --EXPECTED \
  "${P2T_METRICS[@]}" "${P2T_CHECKPOINTS[@]}" "${P2T_PREDICTIONS[@]}" \
  "${P2T_PAIRED_ROWS[@]}" "${P2T_CONFIRMATORY}" \
  "${P2T_NATURAL_METRICS[@]}" "${P2T_NATURAL_PREDICTIONS[@]}" \
  "${P2T_NATURAL_PAIRED_ROWS[@]}" \
  "${EDITOR_SUMMARIES[@]}" "${EDITOR_ROWS[@]}" "${EDITOR_CHECKPOINTS[@]}" \
  "${EDITOR_CONFIRMATORY}"

run_stage validate_complete \
  "${PYTHON}" "${PROJECT_ROOT}/scripts/orchestration/validate_petct_route_a_receipt_pipeline.py" \
  --inputs "${PIPELINE_COMPLETE_INPUTS}" --target complete \
  --output "${RUN_ROOT}/artifacts/ROUTE_A_COMPLETE.json"

PRIMARY_ROUTE_A_COMPLETE="${RUN_ROOT}/artifacts/ROUTE_A_COMPLETE.json"
echo "Route A primary receipt pipeline completed: ${PRIMARY_ROUTE_A_COMPLETE}"

# The all-strategy corpus is a post-completion robustness asset.  It reuses the
# primary residual receipts and never retrains any primary model.  Failure is
# independently receipted and cannot revoke ROUTE_A_COMPLETE.
ROBUSTNESS_ALL_EPISODES="${RUN_ROOT}/artifacts/robustness_all_episodes.jsonl"
NATURAL_ROBUSTNESS_DATA_READY="${RUN_ROOT}/artifacts/NATURAL_ROBUSTNESS_DATA_READY.json"
ROBUSTNESS_ALL_READY="${RUN_ROOT}/artifacts/SCRIBBLE_ROBUSTNESS_ALL_READY.json"
ROBUSTNESS_ALL_FAILED="${RUN_ROOT}/artifacts/SCRIBBLE_ROBUSTNESS_ALL_FAILED.json"
ROBUSTNESS_STATUS=0
if run_stage robustness_all_corpus \
  "${PYTHON}" "${PROJECT_ROOT}/scripts/data/build_petct_scribble_dataset.py" \
  --residual-manifest "${SCOPED_RESIDUALS}" --residual-ready "${RESIDUAL_READY}" \
  --official-simulator "${OFFICIAL_SIMULATOR}" \
  --official-runtime-manifest "${OFFICIAL_RUNTIME_MANIFEST}" \
  --experiment-config "${EXPERIMENT_CONFIG}" --lane natural --oof-ready "${OOF_READY}" \
  --learning-split "${LEARNING_SPLIT}" --partitions "${SELECTED_PARTITIONS[@]}" \
  "${TEST_ACCESS_ARGS[@]}" \
  --strategy-mode all \
  --visible-root "${RUN_ROOT}/artifacts/robustness_all_visible" \
  --evaluation-root "${RUN_ROOT}/artifacts/robustness_all_evaluation" \
  --authorized-root "${RUN_ROOT}/artifacts/robustness_all_authorized" \
  --output-manifest "${ROBUSTNESS_ALL_EPISODES}" \
  --exclusions "${RUN_ROOT}/artifacts/robustness_all_exclusions.jsonl" \
  --ready-receipt "${NATURAL_ROBUSTNESS_DATA_READY}"; then
  ROBUSTNESS_STATUS=0
else
  ROBUSTNESS_STATUS=$?
fi
if [[ ${ROBUSTNESS_STATUS} -eq 0 ]]; then
  ROBUSTNESS_INPUTS="${RUN_ROOT}/artifacts/pipeline_robustness_all_inputs.json"
  if run_stage pipeline_robustness_all_inputs "${PYTHON}" -c '
import json, os, sys
keys = ("experiment_config", "case_manifest", "learning_split", "primary_complete", "primary_natural_episode_manifest", "robustness_natural_episode_manifest")
payload = dict(zip(keys, sys.argv[2:8]))
payload["test_access_receipt"] = sys.argv[8] or None
payload["run_root"] = sys.argv[9]
payload["evaluation_partition"] = sys.argv[10]
payload["natural_robustness_data_ready"] = sys.argv[11]
fd = os.open(sys.argv[1], os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o640)
with os.fdopen(fd, "w", encoding="utf-8") as stream:
    json.dump(payload, stream, indent=2, sort_keys=True); stream.write("\n")
' "${ROBUSTNESS_INPUTS}" "${EXPERIMENT_CONFIG}" "${CASE_MANIFEST}" "${LEARNING_SPLIT}" \
    "${PRIMARY_ROUTE_A_COMPLETE}" "${NATURAL_EPISODES}" "${ROBUSTNESS_ALL_EPISODES}" \
    "${TEST_ACCESS_RECEIPT}" "${RUN_ROOT}" "${EVALUATION_PARTITION}" \
    "${NATURAL_ROBUSTNESS_DATA_READY}"; then
    ROBUSTNESS_STATUS=0
  else
    ROBUSTNESS_STATUS=$?
  fi
fi
if [[ ${ROBUSTNESS_STATUS} -eq 0 ]]; then
  if run_stage validate_robustness_all \
    "${PYTHON}" "${PROJECT_ROOT}/scripts/orchestration/validate_petct_route_a_receipt_pipeline.py" \
    --inputs "${ROBUSTNESS_INPUTS}" --target robustness_all --output "${ROBUSTNESS_ALL_READY}"; then
    ROBUSTNESS_STATUS=0
  else
    ROBUSTNESS_STATUS=$?
  fi
fi
if [[ ${ROBUSTNESS_STATUS} -ne 0 ]]; then
  set +e
  "${PYTHON}" -c '
import hashlib, json, os, sys
output, primary, status = sys.argv[1:]
payload = {
    "schema_version": "PETCT-SCRIBBLE-ROBUSTNESS-ALL-FAILED-v1.0",
    "status": "FAILED",
    "secondary": True,
    "not_in_primary_confirmatory": True,
    "does_not_invalidate_primary_route_a": True,
    "failure_exit_code": int(status),
    "primary_route_a_complete": {
        "path": os.path.abspath(primary),
        "sha256": hashlib.sha256(open(primary, "rb").read()).hexdigest(),
    },
}
fd = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o640)
with os.fdopen(fd, "w", encoding="utf-8") as stream:
    json.dump(payload, stream, indent=2, sort_keys=True); stream.write("\n")
' "${ROBUSTNESS_ALL_FAILED}" "${PRIMARY_ROUTE_A_COMPLETE}" "${ROBUSTNESS_STATUS}"
  set -e
  echo "Robustness-all failed independently: ${ROBUSTNESS_ALL_FAILED}" >&2
else
  echo "Robustness-all receipt ready: ${ROBUSTNESS_ALL_READY}"
fi

# The v2 simple-first campaign ends here. Deferred P2T architectures require a
# future preregistration and are intentionally not entered by this launcher.
echo "Route A six-class simple-first campaign complete: ${PRIMARY_ROUTE_A_COMPLETE}"
exit 0
