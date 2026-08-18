#!/usr/bin/env bash
# Wait for at least one idle GPU (D-2026-08-17-02 single-free-card policy),
# then build the clean M0_V6_FIVEFOLD_OOF on whichever candidate cards are free.
set -euo pipefail

if [[ $# -ne 8 ]]; then
  echo "usage: $0 <state-prefix> <run-root> <ready-json> <source-manifest> <splits-final> <model-root> <gpu0> <gpu1>" >&2
  exit 2
fi

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
STATE=$(readlink -m "$1")
RUN_ROOT=$(readlink -m "$2")
READY=$(readlink -m "$3")
SOURCE_MANIFEST=$(readlink -m "$4")
SPLITS_FINAL=$(readlink -m "$5")
MODEL_ROOT=$(readlink -m "$6")
GPU0="$7"; GPU1="$8"
case "$STATE" in "$PROJECT_ROOT"/nnunet/logs/m0_v6_oof_*) ;; *) echo "invalid state prefix" >&2; exit 2;; esac
for path in "$SOURCE_MANIFEST" "$SPLITS_FINAL" "$MODEL_ROOT/plans.json" "$MODEL_ROOT/dataset.json"; do
  [[ -f "$path" && ! -L "$path" ]] || { echo "missing OOF input: $path" >&2; exit 2; }
done
[[ ! -e "$RUN_ROOT" && ! -L "$RUN_ROOT" && ! -e "$READY" && ! -L "$READY" ]] || {
  echo "OOF output already exists" >&2; exit 2;
}
for suffix in state.json done fail log; do
  [[ ! -e "$STATE.$suffix" && ! -L "$STATE.$suffix" ]] || { echo "queue state already exists" >&2; exit 2; }
done

export TMPDIR=/mnt/HDD4/zlei0805/honor_degree/.tmp
export PIP_CACHE_DIR=/mnt/HDD4/zlei0805/honor_degree/.pip_cache
export CONDA_PKGS_DIRS=/mnt/HDD4/zlei0805/honor_degree/.conda_pkgs
export XDG_CACHE_HOME=/mnt/HDD4/zlei0805/honor_degree/.xdg_cache
export HF_HOME=/mnt/HDD4/zlei0805/honor_degree/.hf_home
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR" "$CONDA_PKGS_DIRS" "$XDG_CACHE_HOME" "$HF_HOME" "$(dirname "$STATE")"

safe=0
free_gpus=()
while [[ "$safe" -lt 3 ]]; do
  free_gpus=()
  for gpu in "$GPU0" "$GPU1"; do
    compute=$(nvidia-smi -i "$gpu" --query-compute-apps=gpu_uuid --format=csv,noheader 2>/dev/null | sed '/^[[:space:]]*$/d' || true)
    temp=$(nvidia-smi -i "$gpu" --query-gpu=temperature.gpu --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
    if [[ -z "$compute" && "$temp" =~ ^[0-9]+$ && "$temp" -lt 85 ]]; then
      free_gpus+=("$gpu")
    fi
  done
  disk_bytes=$(df --output=avail -B1 /mnt/HDD4 | tail -1 | tr -d ' ')
  if [[ "${#free_gpus[@]}" -ge 1 && "$disk_bytes" -ge 21474836480 ]]; then
    safe=$((safe + 1))
  else
    safe=0
  fi
  printf '{"status":"WAITING_GPU","source_m0_lineage":"M0_V6_FIVEFOLD_OOF","safe_gpu_check":%d,"required":3,"free_gpus":"%s","hdd4_available_bytes":%s}\n' "$safe" "${free_gpus[*]-}" "$disk_bytes" > "$STATE.state.json"
  if [[ "$safe" -lt 3 ]]; then sleep 60; fi
done

LAUNCH_GPU0="${free_gpus[0]-}"
LAUNCH_GPU1="${free_gpus[1]--1}"
[[ -n "$LAUNCH_GPU0" ]] || {
  printf '{"status":"FAIL","source_m0_lineage":"M0_V6_FIVEFOLD_OOF","reason":"no_free_gpu_at_launch"}\n' > "$STATE.fail"
  exit 1
}
printf '{"status":"LAUNCHING","source_m0_lineage":"M0_V6_FIVEFOLD_OOF","safe_gpu_check_3_of_3":true,"free_gpus":"%s,%s"}\n' "$LAUNCH_GPU0" "$LAUNCH_GPU1" > "$STATE.state.json"
set +e
"$PROJECT_ROOT/scripts/orchestration/launch_petct_m0_v6_oof.sh" \
  "$RUN_ROOT" "$READY" "$SOURCE_MANIFEST" "$SPLITS_FINAL" "$MODEL_ROOT" "$LAUNCH_GPU0" "$LAUNCH_GPU1" \
  > "$STATE.log" 2>&1
rc=$?
set -e
if [[ "$rc" -eq 0 ]]; then
  printf '{"status":"PASS","source_m0_lineage":"M0_V6_FIVEFOLD_OOF","ready":"%s"}\n' "$READY" > "$STATE.done"
else
  printf '{"status":"FAIL","source_m0_lineage":"M0_V6_FIVEFOLD_OOF","exit_code":%d}\n' "$rc" > "$STATE.fail"
fi
exit "$rc"
