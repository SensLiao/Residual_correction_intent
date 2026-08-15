#!/usr/bin/env bash
set -euo pipefail

# Wait only for the already-running, receipt-bound M0 campaign. This watcher
# never starts, stops, signals, or kills a process. Once all baseline evidence
# and both GPUs are idle, run the val-only Route A chain in the foreground,
# verify its primary completion receipt, and then run nnInteractive serially.

DEFAULT_PROJECT_ROOT="/mnt/HDD4/zlei0805/honor_degree/projects/petct_textual_intent"
DEFAULT_CAMPAIGN_ID="petct_m0_5fold_20260717T121633Z"
DEFAULT_AUDIT_JSON="${DEFAULT_PROJECT_ROOT}/nnunet/audits/psma_v3_nifti_audit_20260717T150905/psma_v3_nifti_audit.json"
DEFAULT_ROUTE_RUN_ROOT="${DEFAULT_PROJECT_ROOT}/route_a/runs/route_a_val_${DEFAULT_CAMPAIGN_ID}"
DEFAULT_EXTERNAL_RUN_ROOT="${DEFAULT_PROJECT_ROOT}/route_a/external/nninteractive_val_${DEFAULT_CAMPAIGN_ID}"
EXPECTED_SIMULATOR_SHA256="a2124e8aa4207e53ac93259214a35b7cf74626f83ab164e519769f86557d7cd2"
EXPECTED_METRICS_SHA256="93e303219deb46b10fc5e5532873a42745aec1ecd6f78335f36cebba62104b83"
EXPECTED_F0_ENV_BUNDLE="87a2261af9d99eb8232a078a2f7ba81cf9f3b4a6389410c296ca9b8671246006"
POLL_SECONDS=120
MAX_WAIT_SECONDS=604800

usage() {
  cat >&2 <<'EOF'
Usage:
  watch_and_run_petct_route_a_after_m0.sh [options]

Path and device overrides:
  --project-root DIR       Remote project root.
  --campaign-id ID         Frozen M0 campaign id.
  --campaign-root DIR      Frozen M0 campaign root.
  --full-ready FILE        FULL_TRAIN_READY.json path.
  --audit-json FILE        PASS PSMA-v3 audit JSON.
  --simulator FILE         Pinned AutoPET V simulate_scribbles.py.
  --metrics FILE           Pinned AutoPET V metrics.py.
  --route-run-root DIR     Fresh Route A output root.
  --external-run-root DIR  Fresh nnInteractive output root.
  --python FILE            Pinned core Python executable.
  --gpu0 ID                Route A first physical GPU (default: 0).
  --gpu1 ID                Route A second and nnInteractive GPU (default: 1).

The evaluation partition is intentionally fixed to val. This watcher exposes
no test-access option and performs no download or process-management action.
EOF
}

PROJECT_ROOT="${DEFAULT_PROJECT_ROOT}"
CAMPAIGN_ID="${DEFAULT_CAMPAIGN_ID}"
CAMPAIGN_ROOT=""
FULL_TRAIN_READY=""
AUDIT_JSON=""
OFFICIAL_SIMULATOR=""
OFFICIAL_METRICS=""
ROUTE_RUN_ROOT=""
EXTERNAL_RUN_ROOT=""
CORE_PYTHON=""
GPU0_ID=0
GPU1_ID=1
while [[ $# -gt 0 ]]; do
  case "$1" in
    --project-root) PROJECT_ROOT="${2:-}"; shift 2 ;;
    --campaign-id) CAMPAIGN_ID="${2:-}"; shift 2 ;;
    --campaign-root) CAMPAIGN_ROOT="${2:-}"; shift 2 ;;
    --full-ready) FULL_TRAIN_READY="${2:-}"; shift 2 ;;
    --audit-json) AUDIT_JSON="${2:-}"; shift 2 ;;
    --simulator) OFFICIAL_SIMULATOR="${2:-}"; shift 2 ;;
    --metrics) OFFICIAL_METRICS="${2:-}"; shift 2 ;;
    --route-run-root) ROUTE_RUN_ROOT="${2:-}"; shift 2 ;;
    --external-run-root) EXTERNAL_RUN_ROOT="${2:-}"; shift 2 ;;
    --python) CORE_PYTHON="${2:-}"; shift 2 ;;
    --gpu0) GPU0_ID="${2:-}"; shift 2 ;;
    --gpu1) GPU1_ID="${2:-}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

CAMPAIGN_ROOT="${CAMPAIGN_ROOT:-${PROJECT_ROOT}/nnunet/full_training_runs/${CAMPAIGN_ID}}"
FULL_TRAIN_READY="${FULL_TRAIN_READY:-${PROJECT_ROOT}/nnunet/manifests/FULL_TRAIN_READY.json}"
AUDIT_JSON="${AUDIT_JSON:-${PROJECT_ROOT}/nnunet/audits/psma_v3_nifti_audit_20260717T150905/psma_v3_nifti_audit.json}"
OFFICIAL_SIMULATOR="${OFFICIAL_SIMULATOR:-${PROJECT_ROOT}/external_runners/autopetv_protocol/interactive/simulate_scribbles.py}"
OFFICIAL_METRICS="${OFFICIAL_METRICS:-${PROJECT_ROOT}/external_runners/autopetv_protocol/metrics.py}"
OFFICIAL_RUNTIME_MANIFEST="${OFFICIAL_RUNTIME_MANIFEST:-${PROJECT_ROOT}/protocols/autopetv_protocol_runtime.json}"
ROUTE_RUN_ROOT="${ROUTE_RUN_ROOT:-${PROJECT_ROOT}/route_a/runs/route_a_val_${CAMPAIGN_ID}}"
EXTERNAL_RUN_ROOT="${EXTERNAL_RUN_ROOT:-${PROJECT_ROOT}/route_a/external/nninteractive_val_${CAMPAIGN_ID}}"
CORE_PYTHON="${CORE_PYTHON:-${PROJECT_ROOT}/envs/petct_nnunet_v281/bin/python}"
EXPERIMENT_CONFIG="${PROJECT_ROOT}/configs/petct_route_a_experiment.json"
CORE_ENV_MARKER="${PROJECT_ROOT}/nnunet/envs/ENV_READY.done"
F0_VALIDATOR="${PROJECT_ROOT}/scripts/orchestration/validate_petct_route_a_f0.py"
F0_READY="${PROJECT_ROOT}/route_a/manifests/F0_READY.json"
F0_TEST_LOG="${PROJECT_ROOT}/route_a/manifests/F0_TESTS.log"

if [[ ! "${CAMPAIGN_ID}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$ ]]; then
  echo "Unsafe campaign id: ${CAMPAIGN_ID}" >&2
  exit 2
fi
if [[ ! "${GPU0_ID}" =~ ^[0-9]+$ || ! "${GPU1_ID}" =~ ^[0-9]+$ || "${GPU0_ID}" == "${GPU1_ID}" ]]; then
  echo "Two distinct numeric GPU ids are required." >&2
  exit 2
fi

FULL_TRAIN_VALIDATOR="${PROJECT_ROOT}/scripts/baseline/validate_petct_m0_full_training.py"
ROUTE_LAUNCHER="${PROJECT_ROOT}/scripts/orchestration/run_petct_route_a_after_baseline.sh"
EXTERNAL_LAUNCHER="${PROJECT_ROOT}/scripts/orchestration/run_petct_external_comparators_after_data.sh"
NNINTERACTIVE_SETUP="${PROJECT_ROOT}/scripts/setup/setup_nninteractive_env.sh"

if [[ ! -d "${PROJECT_ROOT}" || -L "${PROJECT_ROOT}" || ! -d "${CAMPAIGN_ROOT}" || -L "${CAMPAIGN_ROOT}" ]]; then
  echo "Project root or frozen campaign root is missing or symlinked." >&2
  exit 3
fi
for required in "${FULL_TRAIN_VALIDATOR}" "${ROUTE_LAUNCHER}" \
  "${AUDIT_JSON}" "${OFFICIAL_SIMULATOR}" "${OFFICIAL_METRICS}" \
  "${OFFICIAL_RUNTIME_MANIFEST}" "${EXPERIMENT_CONFIG}" \
  "${CORE_ENV_MARKER}" "${F0_VALIDATOR}"; do
  if [[ ! -f "${required}" || -L "${required}" ]]; then
    echo "Missing regular prerequisite: ${required}" >&2
    exit 3
  fi
done
CORE_ENV_PREFIX="${PROJECT_ROOT}/envs/petct_nnunet_v281"
CORE_PYTHON_RESOLVED="$(readlink -f -- "${CORE_PYTHON}" 2>/dev/null || true)"
if [[ -z "${CORE_PYTHON_RESOLVED}" || ! -f "${CORE_PYTHON_RESOLVED}" \
  || ! -x "${CORE_PYTHON_RESOLVED}" \
  || "${CORE_PYTHON_RESOLVED}" != "${CORE_ENV_PREFIX}"/bin/python* ]]; then
  echo "Pinned Python must resolve to an executable inside ${CORE_ENV_PREFIX}: ${CORE_PYTHON}" >&2
  exit 3
fi

validate_fresh_run_roots() {
  if [[ -e "${ROUTE_RUN_ROOT}" || -L "${ROUTE_RUN_ROOT}" ]]; then
    echo "Route A run root must not exist: ${ROUTE_RUN_ROOT}" >&2
    return 1
  fi
  if [[ -e "${EXTERNAL_RUN_ROOT}" || -L "${EXTERNAL_RUN_ROOT}" ]]; then
    echo "External run root must not exist: ${EXTERNAL_RUN_ROOT}" >&2
    return 1
  fi
  "${CORE_PYTHON}" - "${PROJECT_ROOT}" "${ROUTE_RUN_ROOT}" "${EXTERNAL_RUN_ROOT}" <<'PY'
import sys
from pathlib import Path

project = Path(sys.argv[1]).resolve()
roots = [Path(value) for value in sys.argv[2:]]
if any(not root.is_absolute() for root in roots):
    raise SystemExit("run roots must be absolute")
resolved = [root.resolve(strict=False) for root in roots]
if any(not root.is_relative_to(project) for root in resolved):
    raise SystemExit("run roots must remain below the project root")
if resolved[0] == resolved[1]:
    raise SystemExit("Route A and external run roots must differ")
PY
}

# Freshness is checked before waiting and again immediately before each owner
# launcher. The watcher itself never mkdirs either run root.
validate_fresh_run_roots

"${CORE_PYTHON}" - "${AUDIT_JSON}" "${OFFICIAL_SIMULATOR}" \
  "${EXPECTED_SIMULATOR_SHA256}" "${OFFICIAL_METRICS}" "${EXPECTED_METRICS_SHA256}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

audit, simulator, simulator_sha, metrics, metrics_sha = sys.argv[1:]
audit_path = Path(audit)
payload = json.loads(audit_path.read_text(encoding="utf-8"))
if payload.get("status") != "PASS" or payload.get("errors") not in ([], None):
    raise SystemExit("PSMA audit is not a clean PASS")
for value, expected, label in (
    (simulator, simulator_sha, "AutoPET V simulator"),
    (metrics, metrics_sha, "AutoPET V metrics"),
):
    observed = hashlib.sha256(Path(value).read_bytes()).hexdigest()
    if observed != expected:
        raise SystemExit(f"{label} SHA-256 mismatch: {observed}")
print(json.dumps({"status": "PASS", "audit_status": "PASS", "protocol_hashes": "VERIFIED"}))
PY

export PETCT_PSMA_AUDIT_JSON="${AUDIT_JSON}"
export PETCT_AUTOPETV_SIMULATOR="${OFFICIAL_SIMULATOR}"
export PETCT_AUTOPETV_METRICS="${OFFICIAL_METRICS}"
export PETCT_AUTOPETV_RUNTIME_MANIFEST="${OFFICIAL_RUNTIME_MANIFEST}"
export PETCT_CORE_PYTHON="${CORE_PYTHON}"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export VECLIB_MAXIMUM_THREADS=4
export TORCH_NUM_THREADS=4
export PETCT_TORCH_THREADS=4
"${CORE_PYTHON}" - <<'PY'
import torch

if torch.get_num_threads() > 4:
    raise SystemExit(f"PyTorch initialized with {torch.get_num_threads()} CPU threads")
print(f"PyTorch CPU thread cap verified: {torch.get_num_threads()}")
PY

WAIT_STARTED_AT="$(date +%s)"
WAIT_DEADLINE=$((WAIT_STARTED_AT + MAX_WAIT_SECONDS))

poll_or_timeout() {
  local reason="$1" now
  now="$(date +%s)"
  if (( now + POLL_SECONDS > WAIT_DEADLINE )); then
    echo "Timed out after at most seven days while ${reason}." >&2
    return 1
  fi
  echo "${reason}; polling again in ${POLL_SECONDS}s."
  sleep "${POLL_SECONDS}"
}

while [[ ! -f "${FULL_TRAIN_READY}" ]]; do
  if [[ -e "${FULL_TRAIN_READY}" || -L "${FULL_TRAIN_READY}" ]]; then
    echo "FULL_TRAIN_READY exists but is not a regular non-symlink file." >&2
    exit 4
  fi
  poll_or_timeout "Waiting for FULL_TRAIN_READY"
done
if [[ -L "${FULL_TRAIN_READY}" ]]; then
  echo "FULL_TRAIN_READY must not be a symlink." >&2
  exit 4
fi

"${CORE_PYTHON}" - "${FULL_TRAIN_READY}" "${CAMPAIGN_ID}" "${CAMPAIGN_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

ready_path = Path(sys.argv[1])
campaign_id = sys.argv[2]
campaign_root = Path(sys.argv[3]).resolve()
if ready_path.is_symlink() or not ready_path.is_file():
    raise SystemExit("FULL_TRAIN_READY is not a regular non-symlink file")
ready = json.loads(ready_path.read_text(encoding="utf-8"))
expected = {
    "status": "COMMITTED",
    "campaign_id": campaign_id,
    "campaign_root": str(campaign_root),
    "full_training_status": "PASS",
    "full_training_performed": True,
    "folds_completed": [0, 1, 2, 3, 4],
}
for key, value in expected.items():
    if ready.get(key) != value:
        raise SystemExit(f"FULL_TRAIN_READY {key} mismatch")
PY

"${CORE_PYTHON}" "${FULL_TRAIN_VALIDATOR}" validate-campaign "${CAMPAIGN_ROOT}" >/dev/null
for fold in 0 1 2 3 4; do
  action_json="$("${CORE_PYTHON}" "${FULL_TRAIN_VALIDATOR}" fold-action "${CAMPAIGN_ROOT}" "${fold}")"
  action="$("${CORE_PYTHON}" -c 'import json,sys; print(json.loads(sys.argv[1])["action"])' "${action_json}")"
  if [[ "${action}" != "SKIP_VERIFIED" ]]; then
    echo "Fold ${fold} is ${action}, not SKIP_VERIFIED." >&2
    exit 5
  fi
done

# F0 is a deterministic local contract suite, not a scientific experiment.
# Once all five folds are verified, this watcher issues it exactly once when
# absent and then revalidates the immutable receipt.  No Route A run root is
# created until both operations pass.
if [[ ! -e "${F0_READY}" && ! -L "${F0_READY}" ]]; then
  if [[ -e "${F0_TEST_LOG}" || -L "${F0_TEST_LOG}" ]]; then
    echo "F0 test log exists without F0_READY; refusing ambiguous recovery." >&2
    exit 5
  fi
  "${CORE_PYTHON}" "${F0_VALIDATOR}" issue \
    --project-root "${PROJECT_ROOT}" \
    --experiment-config "${EXPERIMENT_CONFIG}" \
    --environment-marker "${CORE_ENV_MARKER}" \
    --official-simulator "${OFFICIAL_SIMULATOR}" \
    --official-metrics "${OFFICIAL_METRICS}" \
    --official-runtime-manifest "${OFFICIAL_RUNTIME_MANIFEST}" \
    --expected-env-bundle "${EXPECTED_F0_ENV_BUNDLE}" \
    --output "${F0_READY}" --test-log "${F0_TEST_LOG}" >/dev/null
fi
if [[ ! -f "${F0_READY}" ]]; then
  echo "F0_READY was not published as a regular file." >&2
  exit 5
fi
if [[ -L "${F0_READY}" ]]; then
  echo "F0_READY must not be a symlink." >&2
  exit 5
fi
"${CORE_PYTHON}" "${F0_VALIDATOR}" validate \
  --receipt "${F0_READY}" --project-root "${PROJECT_ROOT}" \
  --experiment-config "${EXPERIMENT_CONFIG}" \
  --environment-marker "${CORE_ENV_MARKER}" \
  --official-simulator "${OFFICIAL_SIMULATOR}" \
  --official-metrics "${OFFICIAL_METRICS}" \
  --official-runtime-manifest "${OFFICIAL_RUNTIME_MANIFEST}" \
  --expected-env-bundle "${EXPECTED_F0_ENV_BUNDLE}" >/dev/null

gpus_are_idle() {
  local gpu output
  for gpu in "${GPU0_ID}" "${GPU1_ID}"; do
    if ! output="$(nvidia-smi -i "${gpu}" --query-compute-apps=pid --format=csv,noheader,nounits 2>&1)"; then
      echo "nvidia-smi compute-process query failed for GPU ${gpu}: ${output}" >&2
      return 2
    fi
    if grep -Eq '^[[:space:]]*[0-9]+[[:space:]]*$' <<<"${output}"; then
      return 1
    fi
  done
  return 0
}

while true; do
  set +e
  gpus_are_idle
  gpu_status=$?
  set -e
  case "${gpu_status}" in
    0) break ;;
    1) poll_or_timeout "Waiting for both selected GPUs to have no compute process" ;;
    *) exit 6 ;;
  esac
done

validate_fresh_run_roots
echo "Starting foreground val-only Route A run: ${ROUTE_RUN_ROOT}"
set +e
"${ROUTE_LAUNCHER}" --run-root "${ROUTE_RUN_ROOT}" --partition val \
  --gpu0 "${GPU0_ID}" --gpu1 "${GPU1_ID}"
ROUTE_STATUS=$?
set -e
if [[ ${ROUTE_STATUS} -ne 0 ]]; then
  echo "Route A launcher failed with exit code ${ROUTE_STATUS}; external run will not start." >&2
  exit "${ROUTE_STATUS}"
fi

ROUTE_COMPLETE="${ROUTE_RUN_ROOT}/artifacts/ROUTE_A_COMPLETE.json"
"${CORE_PYTHON}" - "${ROUTE_COMPLETE}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if path.is_symlink() or not path.is_file():
    raise SystemExit("ROUTE_A_COMPLETE is not a regular non-symlink file")
receipt = json.loads(path.read_text(encoding="utf-8"))
if receipt.get("schema_version") != "PETCT-ROUTE-A-PIPELINE-RECEIPT-v2.0":
    raise SystemExit("ROUTE_A_COMPLETE schema mismatch")
if receipt.get("status") != "PASS" or receipt.get("target") != "complete":
    raise SystemExit("ROUTE_A_COMPLETE is not a complete PASS")
PY
ROUTE_COMPLETE_SHA256="$("${CORE_PYTHON}" -c 'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest())' "${ROUTE_COMPLETE}")"

record_postbaseline_failure() {
  local failed_stage="$1" exit_code="$2" failure_record="$3"
  local after_hash="" preserved=false
  if [[ -f "${ROUTE_COMPLETE}" && ! -L "${ROUTE_COMPLETE}" ]]; then
    after_hash="$("${CORE_PYTHON}" -c 'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest())' "${ROUTE_COMPLETE}")"
  fi
  if [[ -n "${after_hash}" && "${after_hash}" == "${ROUTE_COMPLETE_SHA256}" ]]; then
    preserved=true
  fi
  "${CORE_PYTHON}" - "${failure_record}" "${CAMPAIGN_ID}" "${failed_stage}" \
    "${exit_code}" "${EXTERNAL_RUN_ROOT}" "${ROUTE_COMPLETE}" \
    "${ROUTE_COMPLETE_SHA256}" "${after_hash}" "${preserved}" <<'PY'
import json
import os
import sys
from datetime import datetime, timezone

(
    output,
    campaign,
    failed_stage,
    exit_code,
    external_root,
    primary,
    before,
    after,
    preserved,
) = sys.argv[1:]
payload = {
    "schema_version": "PETCT-POSTBASELINE-FAILURE-v1.0",
    "status": "FAILED",
    "campaign_id": campaign,
    "failed_stage": failed_stage,
    "exit_code": int(exit_code),
    "external_run_root": external_root,
    "route_a_complete": primary,
    "route_a_complete_sha256_before": before,
    "route_a_complete_sha256_after": after or None,
    "primary_receipt_preserved": preserved == "true",
    "recorded_at": datetime.now(timezone.utc).isoformat(),
    "claim_boundary": "Route A primary remains separate; no external result is inferred.",
}
fd = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o640)
with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
    json.dump(payload, stream, indent=2, sort_keys=True)
    stream.write("\n")
print(json.dumps(payload, sort_keys=True))
PY
  [[ "${preserved}" == "true" ]]
}

# The main launcher was foregrounded and has fully exited. Refuse to start an
# external GPU job if any selected GPU acquired a compute process in between.
set +e
gpus_are_idle
POST_ROUTE_GPU_STATUS=$?
set -e
if [[ ${POST_ROUTE_GPU_STATUS} -ne 0 ]]; then
  echo "A selected GPU is not idle after Route A; preserving the primary receipt and stopping." >&2
  exit 7
fi

# External-environment integrity is intentionally checked only after the Route
# A primary shell has fully exited and ROUTE_A_COMPLETE has been verified. A
# stale external receipt can therefore never prevent the primary run starting.
REBIND_STATUS=0
if [[ ! -f "${NNINTERACTIVE_SETUP}" || -L "${NNINTERACTIVE_SETUP}" ]]; then
  echo "Missing regular nnInteractive environment setup script: ${NNINTERACTIVE_SETUP}" >&2
  REBIND_STATUS=3
else
  set +e
  PETCT_SKIP_INSTALL=1 PIP_NO_INDEX=1 PETCT_NNINTERACTIVE_SMOKE_GPU="${GPU1_ID}" \
    "${NNINTERACTIVE_SETUP}"
  REBIND_STATUS=$?
  set -e
fi
NNINTERACTIVE_READY="${PROJECT_ROOT}/envs/nninteractive_v1.READY.json"
if [[ ${REBIND_STATUS} -eq 0 ]]; then
  set +e
  "${CORE_PYTHON}" - "${NNINTERACTIVE_READY}" \
    "${PROJECT_ROOT}/configs/petct_external_comparators.json" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

ready_path, config_path = map(Path, sys.argv[1:])
if ready_path.is_symlink() or not ready_path.is_file():
    raise SystemExit("nnInteractive READY is not a regular non-symlink file")
ready = json.loads(ready_path.read_text(encoding="utf-8"))
config_sha = hashlib.sha256(config_path.read_bytes()).hexdigest()
expected = {
    "schema_version": "PETCT-NNINTERACTIVE-ENV-v1.1",
    "status": "PASS",
    "config_sha256": config_sha,
    "setup_mode": "VERIFY_EXISTING_NO_INSTALL",
    "model_load_smoke": "PASS",
    "initial_m0_api_smoke": "PASS",
    "scribble_api_smoke": "PASS",
    "adapter_cli_smoke": "PASS",
    "synthetic_only": True,
    "scientific_prediction_produced": False,
    "network_policy_at_runtime": "NO_DOWNLOADS",
}
for key, value in expected.items():
    if ready.get(key) != value:
        raise SystemExit(f"nnInteractive READY {key} mismatch")
PY
  REBIND_STATUS=$?
  set -e
fi
if [[ ${REBIND_STATUS} -ne 0 ]]; then
  record_postbaseline_failure nninteractive_environment_rebind "${REBIND_STATUS}" \
    "${ROUTE_RUN_ROOT}/artifacts/NNINTERACTIVE_ENVIRONMENT_REBIND_FAILED.json" || true
  echo "nnInteractive environment rebind failed with exit code ${REBIND_STATUS}; primary Route A receipt was not edited." >&2
  exit "${REBIND_STATUS}"
fi

# The CUDA smoke process has exited. Reconfirm that both selected GPUs have no
# compute process before the scientific comparator is allowed to start.
set +e
gpus_are_idle
POST_REBIND_GPU_STATUS=$?
set -e
if [[ ${POST_REBIND_GPU_STATUS} -ne 0 ]]; then
  record_postbaseline_failure nninteractive_environment_rebind 7 \
    "${ROUTE_RUN_ROOT}/artifacts/NNINTERACTIVE_ENVIRONMENT_REBIND_FAILED.json" || true
  echo "A selected GPU is not idle after nnInteractive rebind; external execution refused." >&2
  exit 7
fi
if [[ ! -f "${EXTERNAL_LAUNCHER}" || -L "${EXTERNAL_LAUNCHER}" ]]; then
  record_postbaseline_failure nninteractive_external_comparator 3 \
    "${ROUTE_RUN_ROOT}/artifacts/EXTERNAL_NNINTERACTIVE_FAILED.json" || true
  echo "Missing regular external comparator launcher: ${EXTERNAL_LAUNCHER}" >&2
  exit 3
fi
if [[ -e "${EXTERNAL_RUN_ROOT}" || -L "${EXTERNAL_RUN_ROOT}" ]]; then
  echo "External run root appeared before launch: ${EXTERNAL_RUN_ROOT}" >&2
  exit 8
fi

echo "Starting foreground nnInteractive val run on GPU ${GPU1_ID}: ${EXTERNAL_RUN_ROOT}"
set +e
"${EXTERNAL_LAUNCHER}" --route-a-run-root "${ROUTE_RUN_ROOT}" \
  --run-root "${EXTERNAL_RUN_ROOT}" --methods nninteractive --partition val \
  --gpu-nninteractive "${GPU1_ID}"
EXTERNAL_STATUS=$?
set -e

ROUTE_COMPLETE_SHA256_AFTER=""
if [[ -f "${ROUTE_COMPLETE}" && ! -L "${ROUTE_COMPLETE}" ]]; then
  ROUTE_COMPLETE_SHA256_AFTER="$("${CORE_PYTHON}" -c 'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest())' "${ROUTE_COMPLETE}")"
fi
PRIMARY_RECEIPT_PRESERVED=false
if [[ -n "${ROUTE_COMPLETE_SHA256_AFTER}" && "${ROUTE_COMPLETE_SHA256_AFTER}" == "${ROUTE_COMPLETE_SHA256}" ]]; then
  PRIMARY_RECEIPT_PRESERVED=true
fi

if [[ ${EXTERNAL_STATUS} -ne 0 ]]; then
  FAILURE_RECORD="${ROUTE_RUN_ROOT}/artifacts/EXTERNAL_NNINTERACTIVE_FAILED.json"
  record_postbaseline_failure nninteractive_external_comparator "${EXTERNAL_STATUS}" \
    "${FAILURE_RECORD}" || true
  echo "nnInteractive failed with exit code ${EXTERNAL_STATUS}; failure receipt recorded without editing ROUTE_A_COMPLETE." >&2
  exit "${EXTERNAL_STATUS}"
fi
if [[ "${PRIMARY_RECEIPT_PRESERVED}" != "true" ]]; then
  echo "ROUTE_A_COMPLETE changed or disappeared during the external run." >&2
  exit 9
fi

EXTERNAL_COMPLETE="${EXTERNAL_RUN_ROOT}/artifacts/EXTERNAL_COMPARATORS_COMPLETE.json"
"${CORE_PYTHON}" - "${EXTERNAL_COMPLETE}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if path.is_symlink() or not path.is_file():
    raise SystemExit("EXTERNAL_COMPARATORS_COMPLETE is not a regular non-symlink file")
receipt = json.loads(path.read_text(encoding="utf-8"))
if receipt.get("schema_version") != "PETCT-EXTERNAL-COMPARATORS-COMPLETE-v1.0":
    raise SystemExit("external completion schema mismatch")
if receipt.get("status") != "COMPLETE" or receipt.get("selected_methods") != ["nninteractive"]:
    raise SystemExit("nnInteractive completion receipt is incomplete")
PY

echo "Post-baseline Route A and nnInteractive val execution completed with preserved receipts."
