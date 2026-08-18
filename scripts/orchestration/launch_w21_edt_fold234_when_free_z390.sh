#!/usr/bin/env bash
# z390 W2.1 EDT (Dataset903) fold2-4 queue: wait for the RTX 3090 to be free,
# then train folds 2, 3, 4 sequentially. D-2026-08-17-04.
# Single-GPU policy (3090 only; 1080 Ti untouched). TMPDIR fully redirected to
# HDD4 (z390 root partition is at the red line). HDD4 floor = 15G before start.
# Five-fold split = eb56870c... (installed and verified 2026-08-17).
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 <state-prefix> <receipt-json>" >&2
  exit 2
fi

ROOT=/mnt/HDD4/honor_petct
W21="$ROOT/w21"
STATE=$(readlink -m "$1")
RECEIPT=$(readlink -m "$2")
case "$STATE" in "$ROOT"/logs/w21_edt_fold234_*) ;; *) echo "invalid state prefix" >&2; exit 2;; esac
case "$RECEIPT" in "$W21"/artifacts/W21_EDT_FOLD234_DONE.json) ;; *) echo "invalid receipt path" >&2; exit 2;; esac
for suffix in state.json done fail log; do
  [[ ! -e "$STATE.$suffix" && ! -L "$STATE.$suffix" ]] || { echo "queue state already exists" >&2; exit 2; }
done
[[ ! -e "$RECEIPT" && ! -L "$RECEIPT" ]] || { echo "receipt already exists" >&2; exit 2; }

export TMPDIR="$W21/.tmp" PIP_CACHE_DIR="$W21/.pip_cache" CONDA_PKGS_DIRS="$W21/.conda_pkgs"
export XDG_CACHE_HOME="$W21/.xdg_cache" HF_HOME="$W21/.hf_home"
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR" "$CONDA_PKGS_DIRS" "$XDG_CACHE_HOME" "$HF_HOME" "$(dirname "$STATE")"

export nnUNet_raw="$W21/artifacts/nnUNet_raw"
export nnUNet_preprocessed="$W21/artifacts/nnUNet_preprocessed"
export nnUNet_results="$W21/artifacts/nnUNet_results"
export nnUNet_n_proc_DA=6
export OMP_NUM_THREADS=1
export PATH="$ROOT/envs/petct_nnunet_v281/bin:$PATH"

GPU=0
DONE=""
trap 'if [[ -z "$DONE" && ! -e "$STATE.fail" ]]; then printf "{\"status\":\"FAIL\",\"gpu\":%s,\"reason\":\"early_exit\"}\n" "$GPU" > "$STATE.fail"; fi' EXIT

safe=0
while [[ "$safe" -lt 3 ]]; do
  compute=$(nvidia-smi -i "$GPU" --query-compute-apps=gpu_uuid --format=csv,noheader 2>/dev/null | sed '/^[[:space:]]*$/d' || true)
  temp=$(nvidia-smi -i "$GPU" --query-gpu=temperature.gpu --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
  disk_bytes=$(df --output=avail -B1 /mnt/HDD4 | tail -1 | tr -d ' ')
  if [[ -z "$compute" && "$temp" =~ ^[0-9]+$ && "$temp" -lt 85 && "$disk_bytes" -ge 16106127360 ]]; then
    safe=$((safe + 1))
  else
    safe=0
  fi
  printf '{"status":"WAITING_GPU","gpu":%s,"safe_gpu_check":%d,"required":3,"hdd4_available_bytes":%s}\n' "$GPU" "$safe" "$disk_bytes" > "$STATE.state.json"
  if [[ "$safe" -lt 3 ]]; then sleep 60; fi
done

RESULT_DIR="$nnUNet_results/Dataset903_PSMA_W21_ScribbleEDT/nnUNetTrainer__nnUNetPlans__3d_fullres"
printf '{"status":"RUNNING","gpu":%s,"safe_gpu_check_3_of_3":true}\n' "$GPU" > "$STATE.state.json"
for fold in 2 3 4; do
  FINAL="$RESULT_DIR/fold_$fold/checkpoint_final.pth"
  if [[ -f "$FINAL" && ! -L "$FINAL" ]]; then
    printf '{"status":"SKIP","gpu":%s,"fold":%s,"reason":"checkpoint_final_present"}\n' "$GPU" "$fold" >> "$STATE.log"
    continue
  fi
  printf '{"status":"TRAINING","gpu":%s,"fold":%s}\n' "$GPU" "$fold" > "$STATE.state.json"
  CUDA_VISIBLE_DEVICES="$GPU" nice -n 5 nnUNetv2_train 903 3d_fullres "$fold" --npz > "$STATE.fold$fold.log" 2>&1
  [[ -f "$FINAL" && ! -L "$FINAL" ]] || { printf '{"status":"FAIL","gpu":%s,"fold":%s,"reason":"missing_checkpoint_final"}\n' "$GPU" "$fold" > "$STATE.fail"; exit 1; }
  sha256sum "$FINAL" | tee -a "$STATE.log"
  printf '{"status":"FOLD_DONE","gpu":%s,"fold":%s}\n' "$GPU" "$fold" > "$STATE.fold$fold.done"
done

python3 - "$RESULT_DIR" "$RECEIPT" << 'PYEOF' || exit 1
import hashlib
import json
import os
import sys

result_dir, receipt = sys.argv[1], sys.argv[2]
folds = {}
for fold in (2, 3, 4):
    path = os.path.join(result_dir, "fold_%d" % fold, "checkpoint_final.pth")
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    folds[str(fold)] = h.hexdigest()
payload = {
    "dataset": "Dataset903_PSMA_W21_ScribbleEDT",
    "trainer": "nnUNetTrainer__nnUNetPlans__3d_fullres",
    "folds": folds,
    "splits_sha256": "eb56870cd52af38341e1bfb9b2f2a04f3c3602d2bbc004417a8b839ea44e455d",
}
with open(receipt, "w") as fh:
    json.dump(payload, fh, indent=2, sort_keys=True)
print("RECEIPT_WRITTEN", receipt)
PYEOF

printf '{"status":"PASS","gpu":%s,"receipt":"%s"}\n' "$GPU" "$RECEIPT" > "$STATE.done"
DONE=1
