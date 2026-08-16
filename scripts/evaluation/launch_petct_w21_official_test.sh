#!/usr/bin/env bash
set -euo pipefail

RUN_ROOT="/mnt/HDD4/zlei0805/honor_degree/projects/petct_textual_intent/route_a/runs/PETCT-W21-OFFICIAL-TEST-20260809-R1"
PYTHON_BIN="/mnt/HDD4/zlei0805/honor_degree/projects/petct_textual_intent/envs/petct_nnunet_v281/bin/python"
RUNNER="/mnt/HDD4/zlei0805/honor_degree/projects/petct_textual_intent/route_a/scripts/evaluation/run_petct_w21_official_test.py"
RECEIPT="$RUN_ROOT/governance/test-access-receipt.json"
LOG_DIR="$RUN_ROOT/logs"
STATUS_DIR="$RUN_ROOT/status"
LOG_FILE="$LOG_DIR/official-test.log"
LOCK_FILE="$RUN_ROOT/governance/official-test-launch.lock"

mkdir -p "$LOG_DIR" "$STATUS_DIR"
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  echo "another W2.1 official-test launcher holds the run lock" >&2
  exit 73
fi

write_status() {
  local state="$1"
  local detail="$2"
  local selected_gpu="${3:-}"
  local tmp="$STATUS_DIR/state.json.tmp"
  printf '{"state":"%s","detail":"%s","gpu":"%s","updated_utc":"%s"}\n' \
    "$state" "$detail" "$selected_gpu" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >"$tmp"
  mv "$tmp" "$STATUS_DIR/state.json"
}

if [[ -f "$STATUS_DIR/DONE" ]]; then
  echo "W2.1 official TEST is already complete"
  exit 0
fi
if [[ ! -r "$RECEIPT" ]]; then
  echo "consumed W2.1 test receipt is missing" >&2
  exit 74
fi

available_kb=$(df -Pk /mnt/HDD4 | awk 'NR==2 {print $4}')
if [[ -z "$available_kb" || "$available_kb" -lt 104857600 ]]; then
  echo "A6000 /mnt/HDD4 has less than 100 GiB free" >&2
  exit 75
fi

write_status "QUEUED" "waiting_for_three_consecutive_safe_gpu_checks"
selected=""
streak=0
while [[ "$streak" -lt 3 ]]; do
  candidate=""
  while IFS=',' read -r idx mem util temp; do
    idx="${idx//[[:space:]]/}"
    mem="${mem//[[:space:]]/}"
    util="${util//[[:space:]]/}"
    temp="${temp//[[:space:]]/}"
    if [[ "$mem" -lt 1024 && "$util" -lt 10 && "$temp" -lt 80 ]]; then
      candidate="$idx"
      break
    fi
  done < <(nvidia-smi --query-gpu=index,memory.used,utilization.gpu,temperature.gpu --format=csv,noheader,nounits)

  if [[ -n "$candidate" && "$candidate" == "$selected" ]]; then
    streak=$((streak + 1))
  elif [[ -n "$candidate" ]]; then
    selected="$candidate"
    streak=1
  else
    selected=""
    streak=0
  fi
  write_status "QUEUED" "safe_gpu_check_${streak}_of_3" "$selected"
  if [[ "$streak" -lt 3 ]]; then
    sleep 60
  fi
done

export CUDA_VISIBLE_DEVICES="$selected"
started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
write_status "RUNNING_BINARY" "initial_plus_five_corrections" "$selected"

on_exit() {
  local rc=$?
  if [[ "$rc" -eq 0 ]]; then
    write_status "COMPLETE" "binary_edt_and_aggregate_complete" "$selected"
    printf '%s\n' "completed_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >"$STATUS_DIR/DONE"
  else
    write_status "FAILED" "runner_exit_${rc}" "$selected"
  fi
}
trap on_exit EXIT

{
  echo "started_utc=$started_utc"
  echo "selected_physical_gpu=$selected"
  "$PYTHON_BIN" -u "$RUNNER" run --receipt "$RECEIPT" --arm binary --device cuda:0
  write_status "RUNNING_EDT" "initial_plus_five_corrections" "$selected"
  "$PYTHON_BIN" -u "$RUNNER" run --receipt "$RECEIPT" --arm edt --device cuda:0
  write_status "AGGREGATING" "per_case_unnormalized_auc" "$selected"
  "$PYTHON_BIN" -u "$RUNNER" aggregate --receipt "$RECEIPT"
} >>"$LOG_FILE" 2>&1
