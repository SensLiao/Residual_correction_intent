#!/usr/bin/env bash
# Durable TRAIN/VAL chain: clean OOF -> R13 data -> smoke -> full effect wave.
# This watcher never opens locked TEST.
set -euo pipefail

if [[ $# -ne 12 ]]; then
  echo "usage: $0 <state-prefix> <oof-state-prefix> <r13-root> <smoke-root> <val-root> <case-manifest> <learning-split> <config> <simulator> <runtime-manifest> <gpu0> <gpu1>" >&2
  exit 2
fi
PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
STATE=$(readlink -m "$1"); OOF_STATE=$(readlink -m "$2")
R13_ROOT=$(readlink -m "$3"); SMOKE_ROOT=$(readlink -m "$4"); VAL_ROOT=$(readlink -m "$5")
CASE_MANIFEST=$(readlink -m "$6"); LEARNING_SPLIT=$(readlink -m "$7"); CONFIG=$(readlink -m "$8")
SIMULATOR=$(readlink -m "$9"); RUNTIME=$(readlink -m "${10}"); GPU0="${11}"; GPU1="${12}"
OOF_READY="$PROJECT_ROOT/nnunet/manifests/M0_V6_FIVEFOLD_OOF_READY.json"
case "$STATE" in "$PROJECT_ROOT"/nnunet/logs/r13_pipeline_*) ;; *) exit 2;; esac
for path in "$CASE_MANIFEST" "$LEARNING_SPLIT" "$CONFIG" "$SIMULATOR" "$RUNTIME"; do
  [[ -f "$path" && ! -L "$path" ]] || exit 2
done
for path in "$R13_ROOT" "$SMOKE_ROOT" "$VAL_ROOT"; do [[ ! -e "$path" && ! -L "$path" ]] || exit 2; done
for suffix in state.json done fail log; do [[ ! -e "$STATE.$suffix" && ! -L "$STATE.$suffix" ]] || exit 2; done

export TMPDIR=/mnt/HDD4/workspace/honor_degree/.tmp
export PIP_CACHE_DIR=/mnt/HDD4/workspace/honor_degree/.pip_cache
export CONDA_PKGS_DIRS=/mnt/HDD4/workspace/honor_degree/.conda_pkgs
export XDG_CACHE_HOME=/mnt/HDD4/workspace/honor_degree/.xdg_cache
export HF_HOME=/mnt/HDD4/workspace/honor_degree/.hf_home
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR" "$CONDA_PKGS_DIRS" "$XDG_CACHE_HOME" "$HF_HOME" "$(dirname "$STATE")"
DONE=""; STAGE="wait_oof"
trap 'if [[ -z "$DONE" ]]; then printf "{\"status\":\"FAIL\",\"stage\":\"%s\",\"source_m0_lineage\":\"M0_V6_FIVEFOLD_OOF\"}\n" "$STAGE" > "$STATE.fail"; fi' EXIT

while [[ ! -f "$OOF_READY" ]]; do
  if [[ -f "$OOF_STATE.fail" ]]; then exit 1; fi
  printf '{"status":"WAITING_OOF","stage":"%s","source_m0_lineage":"M0_V6_FIVEFOLD_OOF"}\n' "$STAGE" > "$STATE.state.json"
  sleep 60
done

STAGE="r13_data"
printf '{"status":"RUNNING","stage":"%s"}\n' "$STAGE" > "$STATE.state.json"
"$PROJECT_ROOT/scripts/orchestration/launch_petct_r13_mainline.sh" \
  --run-root "$R13_ROOT" --oof-ready "$OOF_READY" --case-manifest "$CASE_MANIFEST" \
  --learning-split "$LEARNING_SPLIT" --experiment-config "$CONFIG" \
  --official-simulator "$SIMULATOR" --official-runtime-manifest "$RUNTIME" \
  > "$STATE.r13-data.log" 2>&1

wait_safe_gpus() {
  local safe=0 compute temperatures temperature_ok disk_bytes
  while [[ "$safe" -lt 3 ]]; do
    compute=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader 2>/dev/null | sed '/^[[:space:]]*$/d' || true)
    temperatures=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits)
    disk_bytes=$(df --output=avail -B1 /mnt/HDD4 | tail -1 | tr -d ' ')
    temperature_ok=1
    while IFS= read -r value; do [[ "$value" -lt 85 ]] || temperature_ok=0; done <<< "$temperatures"
    if [[ -z "$compute" && "$temperature_ok" -eq 1 && "$disk_bytes" -ge 21474836480 ]]; then safe=$((safe + 1)); else safe=0; fi
    printf '{"status":"WAITING_GPU","stage":"%s","safe_gpu_check":%d,"safe_gpu_check_3_of_3":%s}\n' "$STAGE" "$safe" "$([[ "$safe" -eq 3 ]] && echo true || echo false)" > "$STATE.state.json"
    if [[ "$safe" -lt 3 ]]; then sleep 60; fi
  done
}

STAGE="effect_smoke"; wait_safe_gpus
"$PROJECT_ROOT/scripts/orchestration/launch_petct_r13_effect_smoke.sh" \
  "$R13_ROOT" "$SMOKE_ROOT" "$GPU0" "$GPU1" > "$STATE.smoke.log" 2>&1
[[ -f "$SMOKE_ROOT/state/smoke.done" ]] || exit 1

STAGE="effect_val"; wait_safe_gpus
"$PROJECT_ROOT/scripts/orchestration/launch_petct_r13_effect_val.sh" \
  "$R13_ROOT" "$VAL_ROOT" "$GPU0" "$GPU1" > "$STATE.effect-val.log" 2>&1
[[ -f "$VAL_ROOT/state/effect_val.done" ]] || exit 1

printf '{"status":"PASS","source_m0_lineage":"M0_V6_FIVEFOLD_OOF","r13_data":"%s","smoke":"%s","effect_val":"%s"}\n' "$R13_ROOT" "$SMOKE_ROOT" "$VAL_ROOT" > "$STATE.done"
rm -f "$STATE.state.json"
DONE=1
