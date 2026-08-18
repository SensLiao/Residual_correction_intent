#!/usr/bin/env bash
# z390 W2.1 fold1 queue (Director 2026-08-18: W2.1 runs on 3090 only, not A6000).
# Waits for the EDT fold2-4 queue receipt, then resumes fold1 for EDT (903) and
# Binary (902) sequentially on the single 3090 with the frozen -c recipe.
# Data prerequisite: TRANSFER_RECEIPT.json in artifacts/.transfer/ (A6000 relay).
# Never consumes locked TEST; training only.
set -euo pipefail

ROOT=/mnt/HDD4/honor_petct
W21="$ROOT/w21"
STATE="$ROOT/logs/w21_fold1_after_fold234_z390"
RECEIPT="$W21/artifacts/W21_FOLD1_AFTER_FOLD234_DONE.json"
FOLD234_DONE="$W21/artifacts/W21_EDT_FOLD234_DONE.json"
TRANSFER="$W21/artifacts/.transfer/TRANSFER_RECEIPT.json"

for suffix in state.json done fail log; do
  [[ ! -e "$STATE.$suffix" && ! -L "$STATE.$suffix" ]] || exit 2
done
[[ ! -e "$RECEIPT" && ! -L "$RECEIPT" ]] || exit 2

export TMPDIR="$W21/.tmp" PIP_CACHE_DIR="$W21/.pip_cache" CONDA_PKGS_DIRS="$W21/.conda_pkgs"
export XDG_CACHE_HOME="$W21/.xdg_cache" HF_HOME="$W21/.hf_home"
mkdir -p "$TMPDIR" "$ROOT/logs"
export nnUNet_raw="$W21/artifacts/nnUNet_raw"
export nnUNet_preprocessed="$W21/artifacts/nnUNet_preprocessed"
export nnUNet_results="$W21/artifacts/nnUNet_results"
export nnUNet_n_proc_DA=6
export OMP_NUM_THREADS=1
export PATH="$ROOT/envs/petct_nnunet_v281/bin:$PATH"

GPU=0
DONE=""
trap 'if [[ -z "$DONE" && ! -e "$STATE.fail" ]]; then printf "{\"status\":\"FAIL\",\"reason\":\"early_exit\"}\n" > "$STATE.fail"; fi' EXIT
fail() { printf '{"status":"FAIL","reason":"%s"}\n' "$1" > "$STATE.fail"; exit 1; }

while [[ ! -f "$FOLD234_DONE" ]]; do
  printf '{"status":"WAITING_FOLD234","required":"W21_EDT_FOLD234_DONE.json"}\n' > "$STATE.state.json"
  sleep 600
done
[[ -f "$TRANSFER" ]] || fail "transfer_receipt_missing"

wait_safe_gpu() {
  local safe=0 mem util temp
  while (( safe < 3 )); do
    IFS=',' read -r mem util temp < <(
      nvidia-smi -i "$GPU" --query-gpu=memory.used,utilization.gpu,temperature.gpu \
        --format=csv,noheader,nounits | tr -d ' '
    )
    if (( mem < 1024 && util <= 10 && temp < 85 )); then
      safe=$((safe + 1))
    else
      safe=0
    fi
    sleep 60
  done
}

resume_arm() {
  local ds="$1"
  local final="$nnUNet_results/$ds/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_1/checkpoint_final.pth"
  if [[ -f "$final" && ! -L "$final" ]]; then
    printf '{"status":"SKIP","ds":"%s","reason":"final_present"}\n' "$ds" >> "$STATE.log"
    return 0
  fi
  wait_safe_gpu
  printf '{"status":"TRAINING","ds":"%s","fold":1,"resume":"-c"}\n' "$ds" > "$STATE.state.json"
  CUDA_VISIBLE_DEVICES="$GPU" nice -n 5 nnUNetv2_train "$ds" 3d_fullres 1 -c --npz \
    > "$ROOT/logs/w21_${ds}_fold1_z390_resume.log" 2>&1
  [[ -f "$final" ]] || fail "${ds}_fold1_no_final"
  sha256sum "$final" | tee -a "$STATE.log"
}

printf '{"status":"RUNNING","fold234_receipt_consumed":true}\n' > "$STATE.state.json"
resume_arm Dataset903_PSMA_W21_ScribbleEDT
resume_arm Dataset902_PSMA_W21_ScribbleBinary

python3 - "$RECEIPT" << 'PYEOF' || fail "receipt_write_failed"
import hashlib
import json
import sys

receipt = sys.argv[1]
folds = {}
for ds in ("Dataset903_PSMA_W21_ScribbleEDT", "Dataset902_PSMA_W21_ScribbleBinary"):
    p = ("/mnt/HDD4/honor_petct/w21/artifacts/nnUNet_results/%s/"
         "nnUNetTrainer__nnUNetPlans__3d_fullres/fold_1/checkpoint_final.pth") % ds
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    folds[ds] = h.hexdigest()
with open(receipt, "w") as fh:
    json.dump({"schema_version": "PETCT-W21-FOLD1-Z390-v1.0", "status": "PASS",
               "fold": 1, "resume_recipe": "nnUNetv2_train <ds> 3d_fullres 1 -c --npz",
               "folds": folds, "locked_test": "zero_contact"}, fh, indent=2, sort_keys=True)
PYEOF

printf '{"status":"PASS","receipt":"%s"}\n' "$RECEIPT" > "$STATE.done"
DONE=1
