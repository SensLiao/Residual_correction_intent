#!/usr/bin/env bash
# Overnight queue (Director 2026-08-18 shutdown instruction): after the B1
# gold-call ceiling pass finishes, resume the paused W2.1 fold1 training on
# both A6000 GPUs with the frozen resume recipe (-c). D-2026-08-17-03/04.
# Never consumes locked TEST; training only. Fail-safe: any gold .fail or a
# W21 terminal marker aborts this queue without starting anything new.
set -euo pipefail

P=/mnt/HDD4/workspace/honor_degree/projects/petct_textual_intent
GOLD=$P/route_a/runs/PETCT-R13-GOLD-CEILING-20260818-R1
RUN=$P/route_a/runs/PETCT-W21-20260805-R1
W=$RUN/artifacts
STATE=$RUN/logs/w21_fold1_resume_overnight_20260818
QUEUE_LOG=$RUN/logs/w21_fold1_resume_overnight.log

export nnUNet_raw=$W/nnUNet_raw
export nnUNet_preprocessed=$W/nnUNet_preprocessed
export nnUNet_results=$W/nnUNet_results
export TMPDIR=/mnt/HDD4/workspace/.tmp
export LIBRARY_PATH=/mnt/HDD4/workspace/.local/lib:${LIBRARY_PATH:-}
export TORCHINDUCTOR_CACHE_DIR=/mnt/HDD4/workspace/.tmp/inductor_w21_5fold
export PATH=$P/envs/petct_nnunet_v281/bin:$PATH
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export nnUNet_n_proc_DA=3

mkdir -p "$RUN/logs" "$TMPDIR"
[[ ! -e "$STATE.state.json" && ! -e "$STATE.done" && ! -e "$STATE.fail" ]] || exit 2
printf '{"status":"WAITING_GOLD_CEILING","required":"gold_call_ceiling.done"}\n' > "$STATE.state.json"

fail() {
  printf '{"status":"FAIL","reason":"%s"}\n' "$1" > "$STATE.fail"
  echo "FAIL $(date -Is) $1" >> "$QUEUE_LOG"
  exit 1
}

while [[ ! -e "$GOLD/state/gold_call_ceiling.done" ]]; do
  [[ ! -e "$GOLD/state/gold_call_ceiling.fail" ]] || fail "gold_ceiling_failed"
  sleep 300
done
echo "GOLD_DONE $(date -Is)" >> "$QUEUE_LOG"

wait_safe_gpu() {
  local gpu="$1" safe=0 mem util temp
  while (( safe < 3 )); do
    IFS=',' read -r mem util temp < <(
      nvidia-smi -i "$gpu" --query-gpu=memory.used,utilization.gpu,temperature.gpu \
        --format=csv,noheader,nounits | tr -d ' '
    )
    if (( mem < 1024 && util <= 10 && temp < 85 )); then
      safe=$((safe + 1))
    else
      safe=0
      echo "WAIT_GPU $(date -Is) gpu$gpu mem${mem}MiB util${util} temp${temp}" >> "$QUEUE_LOG"
    fi
    sleep 60
  done
}

resume_arm() {
  local ds="$1" gpu="$2"
  local final="$W/nnUNet_results/$ds/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_1/checkpoint_final.pth"
  if [[ -f "$final" && ! -L "$final" ]]; then
    echo "SKIP $(date -Is) $ds fold1 already_final" >> "$QUEUE_LOG"
    return 0
  fi
  wait_safe_gpu "$gpu"
  echo "START $(date -Is) $ds fold1 resume gpu$gpu" >> "$QUEUE_LOG"
  CUDA_VISIBLE_DEVICES=$gpu nnUNetv2_train "$ds" 3d_fullres 1 -c --npz \
    >> "$RUN/logs/w21_${ds}_fold1_resume.log" 2>&1
  [[ -f "$final" ]] || { echo "NO_FINAL $(date -Is) $ds" >> "$QUEUE_LOG"; return 1; }
  sha256sum "$final" | tee -a "$QUEUE_LOG"
  echo "DONE $(date -Is) $ds fold1" >> "$QUEUE_LOG"
}

printf '{"status":"RUNNING","stage":"fold1_resume_both_arms"}\n' > "$STATE.state.json"
resume_arm Dataset902_PSMA_W21_ScribbleBinary 0 &
pid0=$!
resume_arm Dataset903_PSMA_W21_ScribbleEDT 1 &
pid1=$!
wait "$pid0" || { kill "$pid1" 2>/dev/null || true; fail "binary_fold1_resume_failed"; }
wait "$pid1" || fail "edt_fold1_resume_failed"

printf '{"status":"PASS","arms":["Dataset902_Binary","Dataset903_EDT"],"fold":1,"resume_recipe":"nnUNetv2_train <ds> 3d_fullres 1 -c --npz","locked_test":"zero_contact"}\n' > "$STATE.done"
rm -f "$STATE.state.json"
echo "QUEUE_DONE $(date -Is)" >> "$QUEUE_LOG"
