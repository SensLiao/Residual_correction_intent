#!/usr/bin/env bash
set -euo pipefail

# Continue an already-running fold-0 campaign without relaunching fold 0.
#
# This entrypoint is intentionally PLAN_ONLY by default.  Execution requires
# both --execute and the campaign-bound confirmation token printed in the plan.
# The two workers are independent, but each GPU queue is strictly serial:
#
#   GPU 0: wait for a *verified* fold-0 receipt -> fold 2 -> fold 4
#   GPU 1: fold 1 -> fold 3 (may begin while fold 0 is still running)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
  cat >&2 <<'EOF'
Usage:
  launch_petct_m0_parallel_continuation.sh <campaign-id> [options]

Options:
  --gpu0 ID             Physical GPU used by fold 0 and the 2 -> 4 queue (default: 0)
  --gpu1 ID             Physical GPU used by the 1 -> 3 queue (default: 1)
  --poll-seconds N      Fold receipt polling interval (default: 120)
  --wait-seconds N      Maximum wait for an already-running fold (default: 604800)
  --execute             Permit execution; still requires --confirm
  --confirm TOKEN       Must equal EXECUTE_PETCT_M0_CONTINUATION_<campaign-id>
  -h, --help            Show this help

Without --execute this command only prints the plan and does not source the
server environment, create locks, launch folds, or publish a manifest.
EOF
}

if [[ $# -lt 1 ]]; then
  usage
  exit 2
fi

CAMPAIGN_ID="$1"
shift
GPU0_ID=0
GPU1_ID=1
POLL_SECONDS=120
WAIT_SECONDS=604800
EXECUTE=false
CONFIRM_TOKEN=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu0) GPU0_ID="${2:-}"; shift 2 ;;
    --gpu1) GPU1_ID="${2:-}"; shift 2 ;;
    --poll-seconds) POLL_SECONDS="${2:-}"; shift 2 ;;
    --wait-seconds) WAIT_SECONDS="${2:-}"; shift 2 ;;
    --execute) EXECUTE=true; shift ;;
    --confirm) CONFIRM_TOKEN="${2:-}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ ! "${CAMPAIGN_ID}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$ ]]; then
  echo "Unsafe campaign id" >&2
  exit 2
fi
if [[ ! "${GPU0_ID}" =~ ^[0-9]+$ || ! "${GPU1_ID}" =~ ^[0-9]+$ ]]; then
  echo "GPU ids must be non-negative integers" >&2
  exit 2
fi
if [[ "${GPU0_ID}" == "${GPU1_ID}" ]]; then
  echo "Two distinct GPU ids are required" >&2
  exit 2
fi
if [[ ! "${POLL_SECONDS}" =~ ^[1-9][0-9]*$ || ! "${WAIT_SECONDS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "Polling and wait durations must be positive integers" >&2
  exit 2
fi

EXPECTED_TOKEN="EXECUTE_PETCT_M0_CONTINUATION_${CAMPAIGN_ID}"
cat <<EOF
status: PLAN_ONLY
campaign: ${CAMPAIGN_ID}
fold 0: do not relaunch; observe existing run on physical GPU ${GPU0_ID}
physical GPU ${GPU1_ID}: fold 1 -> fold 3 (start independently of fold 0)
physical GPU ${GPU0_ID}: wait for fold 0 verified receipt -> fold 2 -> fold 4
within-GPU concurrency: forbidden (one serial queue per GPU)
cross-GPU concurrency: enabled (the two queues run in parallel)
publish gate: every fold-action for 0,1,2,3,4 must be SKIP_VERIFIED
confirmation token: ${EXPECTED_TOKEN}
EOF

if [[ "${EXECUTE}" != "true" ]]; then
  echo "launch_performed: false"
  exit 0
fi
if [[ "${CONFIRM_TOKEN}" != "${EXPECTED_TOKEN}" ]]; then
  echo "Execution refused: pass --confirm ${EXPECTED_TOKEN}" >&2
  exit 6
fi

# Sourcing this file creates standard experiment directories.  Keep it strictly
# below the explicit execution gate so PLAN_ONLY remains filesystem read-only.
# shellcheck source=petct_m0_common.sh
source "${SCRIPT_DIR}/petct_m0_common.sh"

VALIDATOR="${SCRIPT_DIR}/validate_petct_m0_full_training.py"
FOLD_RUNNER="${SCRIPT_DIR}/run_petct_m0_fold.sh"
CAMPAIGN_ROOT="${EXP_ROOT}/full_training_runs/${CAMPAIGN_ID}"
FULL_TRAIN_READY="${EXP_ROOT}/manifests/FULL_TRAIN_READY.json"
LOCK_ROOT="${EXP_ROOT}/locks"

for required in "${PYTHON}" "${VALIDATOR}" "${FOLD_RUNNER}"; do
  if [[ ! -f "${required}" && ! -x "${required}" ]]; then
    echo "Missing continuation prerequisite: ${required}" >&2
    exit 3
  fi
done
if [[ ! -d "${CAMPAIGN_ROOT}" ]]; then
  echo "Campaign does not exist; initialize it with the full-training launcher first" >&2
  exit 3
fi
if ! command -v flock >/dev/null 2>&1; then
  echo "Missing required flock executable" >&2
  exit 3
fi

mkdir -p "${LOCK_ROOT}"
exec 9>"${LOCK_ROOT}/m0_parallel_continuation.lock"
if ! flock -n 9; then
  echo "Another PET/CT M0 continuation launcher is active" >&2
  exit 5
fi

"${PYTHON}" "${VALIDATOR}" validate-campaign "${CAMPAIGN_ROOT}" >/dev/null
if [[ -e "${FULL_TRAIN_READY}" || -L "${FULL_TRAIN_READY}" ]]; then
  echo "FULL_TRAIN_READY already exists; refusing to overwrite or republish it" >&2
  exit 4
fi

fold_action() {
  local fold="$1"
  local payload
  payload="$("${PYTHON}" "${VALIDATOR}" fold-action "${CAMPAIGN_ROOT}" "${fold}")"
  "${PYTHON}" -c \
    'import json,sys; print(json.loads(sys.argv[1])["action"])' "${payload}"
}

fold_lock_is_held() {
  local fold="$1"
  local lock_path="${CAMPAIGN_ROOT}/locks/fold_${fold}.lock"
  [[ -e "${lock_path}" && ! -L "${lock_path}" ]] || return 1
  (
    exec 7>"${lock_path}"
    if flock -n 7; then
      return 1
    fi
    return 0
  )
}

wait_for_verified_fold() {
  local fold="$1"
  local elapsed=0
  local action
  while true; do
    action="$(fold_action "${fold}")"
    if [[ "${action}" == "SKIP_VERIFIED" ]]; then
      echo "Fold ${fold} has a revalidated committed receipt."
      return 0
    fi
    if [[ "${action}" != "FRESH" && "${action}" != "RESUME" ]]; then
      echo "Fold ${fold} returned unsupported action ${action}" >&2
      return 1
    fi
    if ! fold_lock_is_held "${fold}"; then
      echo "Fold ${fold} is not verified and no runner owns its fold lock" >&2
      return 1
    fi
    if (( elapsed >= WAIT_SECONDS )); then
      echo "Timed out waiting for fold ${fold} to publish a verified receipt" >&2
      return 1
    fi
    echo "Fold ${fold} action=${action}; active runner observed, waiting ${POLL_SECONDS}s."
    sleep "${POLL_SECONDS}"
    elapsed=$((elapsed + POLL_SECONDS))
  done
}

run_fold_to_verified() {
  local fold="$1"
  local gpu_id="$2"
  local action rc
  action="$(fold_action "${fold}")"
  if [[ "${action}" == "SKIP_VERIFIED" ]]; then
    echo "Fold ${fold} already verified; idempotent skip."
    return 0
  fi
  if [[ "${action}" != "FRESH" && "${action}" != "RESUME" ]]; then
    echo "Fold ${fold} returned unsupported action ${action}" >&2
    return 1
  fi

  set +e
  "${FOLD_RUNNER}" "${CAMPAIGN_ID}" "${fold}" "${gpu_id}"
  rc=$?
  set -e
  if [[ ${rc} -eq 5 ]] && fold_lock_is_held "${fold}"; then
    echo "Fold ${fold} is owned by another idempotent launcher; following its receipt."
    wait_for_verified_fold "${fold}"
    return $?
  fi
  if [[ ${rc} -ne 0 ]]; then
    echo "Fold ${fold} runner failed with exit code ${rc}" >&2
    return "${rc}"
  fi
  action="$(fold_action "${fold}")"
  if [[ "${action}" != "SKIP_VERIFIED" ]]; then
    echo "Fold ${fold} runner exited without a verified receipt" >&2
    return 1
  fi
}

run_gpu1_queue() {
  # This queue intentionally does not depend on fold 0.
  run_fold_to_verified 1 "${GPU1_ID}"
  run_fold_to_verified 3 "${GPU1_ID}"
}

run_gpu0_queue() {
  # Fold 0 is never launched here.  It must be externally active or verified.
  wait_for_verified_fold 0
  run_fold_to_verified 2 "${GPU0_ID}"
  run_fold_to_verified 4 "${GPU0_ID}"
}

# Start the independent GPU-1 queue first so fold 1 does not wait for fold 0.
run_gpu1_queue &
GPU1_WORKER=$!
run_gpu0_queue &
GPU0_WORKER=$!

set +e
wait "${GPU1_WORKER}"
GPU1_RC=$?
wait "${GPU0_WORKER}"
GPU0_RC=$?
set -e
if [[ ${GPU0_RC} -ne 0 || ${GPU1_RC} -ne 0 ]]; then
  echo "At least one serial GPU queue failed; FULL_TRAIN_READY will not be published" >&2
  exit 7
fi

# A process exit is not evidence.  Re-run the receipt verifier for every fold
# before the validator is allowed to publish the five-fold handoff manifest.
for fold in 0 1 2 3 4; do
  ACTION="$(fold_action "${fold}")"
  if [[ "${ACTION}" != "SKIP_VERIFIED" ]]; then
    echo "Publish refused: fold ${fold} is ${ACTION}, not SKIP_VERIFIED" >&2
    exit 8
  fi
done

"${PYTHON}" "${VALIDATOR}" publish-full-ready \
  "${CAMPAIGN_ROOT}" "${FULL_TRAIN_READY}" \
  --oof-root "${EXP_ROOT}/oof_predictions" \
  --oof-root "${EXP_ROOT}/oof_probabilities" \
  --result-root "${EXP_ROOT}/evaluation"
echo "Five verified folds published: ${FULL_TRAIN_READY}"

