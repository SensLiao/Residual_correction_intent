#!/usr/bin/env bash
# Wait for the full R13 VAL wave, then execute the pre-authorized exact cleanup plan.
set -euo pipefail

if [[ $# -ne 7 ]]; then
  echo "usage: $0 <state-prefix> <pipeline-state-prefix> <r13-root> <smoke-root> <effect-val-root> <capsule-root> <cleanup-receipt>" >&2
  exit 2
fi
PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
STATE=$(readlink -m "$1"); PIPELINE_STATE=$(readlink -m "$2")
R13_ROOT=$(readlink -m "$3"); SMOKE_ROOT=$(readlink -m "$4"); VAL_ROOT=$(readlink -m "$5")
CAPSULE=$(readlink -m "$6"); RECEIPT=$(readlink -m "$7")
PLAN="$PROJECT_ROOT/records/r13-server-folder-classification-20260817.json"
CLEANUP="$PROJECT_ROOT/scripts/orchestration/cleanup_petct_legacy_after_r13.py"
PY="$PROJECT_ROOT/envs/petct_nnunet_v281/bin/python"
for path in "$PLAN" "$CLEANUP"; do [[ -f "$path" && ! -L "$path" ]] || exit 2; done
for path in "$STATE.state.json" "$STATE.done" "$STATE.fail" "$CAPSULE" "$RECEIPT"; do [[ ! -e "$path" && ! -L "$path" ]] || exit 2; done
mkdir -p "$(dirname "$STATE")"
DONE=""
trap 'if [[ -z "$DONE" ]]; then printf "{\"status\":\"FAIL\",\"stage\":\"cleanup_wait_or_execute\"}\n" > "$STATE.fail"; fi' EXIT

while [[ ! -f "$VAL_ROOT/state/effect_val.done" ]]; do
  [[ ! -f "$PIPELINE_STATE.fail" ]] || exit 1
  printf '{"status":"WAITING_EFFECT_VAL","required":"effect_val.done"}\n' > "$STATE.state.json"
  sleep 120
done

printf '{"status":"CLEANING","confirmation":"DELETE_SUPERSEDED_R13_LEGACY"}\n' > "$STATE.state.json"
"$PY" "$CLEANUP" --project-root "$PROJECT_ROOT" --plan "$PLAN" \
  --r13-root "$R13_ROOT" --smoke-root "$SMOKE_ROOT" --capsule-root "$CAPSULE" \
  --receipt "$RECEIPT" --execute-confirmation DELETE_SUPERSEDED_R13_LEGACY \
  > "$STATE.log" 2>&1
test -s "$RECEIPT"
printf '{"status":"PASS","receipt":"%s","capsule":"%s"}\n' "$RECEIPT" "$CAPSULE" > "$STATE.done"
rm -f "$STATE.state.json"
DONE=1
