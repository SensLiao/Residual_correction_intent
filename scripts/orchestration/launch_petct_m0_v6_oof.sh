#!/usr/bin/env bash
# Produce the explicit clean 506-case M0-v6 OOF source for R13-main.
set -euo pipefail

if [[ $# -ne 7 ]]; then
  echo "usage: $0 <fresh-run-root> <fresh-ready-json> <source-manifest> <splits-final> <model-root> <gpu0> <gpu1>" >&2
  exit 2
fi

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
RUN_BASE="$PROJECT_ROOT/nnunet/oof_v6_runs"
RUN_ROOT=$(readlink -m "$1")
READY=$(readlink -m "$2")
SOURCE_MANIFEST=$(readlink -m "$3")
SPLITS_FINAL=$(readlink -m "$4")
MODEL_ROOT=$(readlink -m "$5")
GPU0="$6"
GPU1="$7"
case "$RUN_ROOT" in "$RUN_BASE"/PETCT-M0-V6-OOF-*) ;; *) echo "invalid OOF run root" >&2; exit 2;; esac
case "$READY" in "$PROJECT_ROOT"/nnunet/manifests/M0_V6_FIVEFOLD_OOF_READY.json) ;; *) echo "invalid OOF receipt path" >&2; exit 2;; esac
if [[ ! "$GPU0" =~ ^[0-9]+$ ]]; then
  echo "gpu0 must be a numeric GPU ID" >&2
  exit 2
fi
if [[ "$GPU1" != "-1" ]]; then
  if [[ "$GPU0" == "$GPU1" || ! "$GPU1" =~ ^[0-9]+$ ]]; then
    echo "two distinct numeric GPU IDs (or gpu1=-1 for single-GPU mode) are required" >&2
    exit 2
  fi
fi
for path in "$SOURCE_MANIFEST" "$SPLITS_FINAL" "$MODEL_ROOT/plans.json" "$MODEL_ROOT/dataset.json"; do
  [[ -f "$path" && ! -L "$path" ]] || { echo "missing explicit OOF input: $path" >&2; exit 2; }
done
[[ ! -e "$RUN_ROOT" && ! -L "$RUN_ROOT" && ! -e "$READY" && ! -L "$READY" ]] || {
  echo "OOF output already exists" >&2; exit 2;
}

PY="$PROJECT_ROOT/envs/petct_nnunet_v281/bin/python"
SCRIPT="$PROJECT_ROOT/scripts/baseline/run_petct_m0_v6_oof.py"
export TMPDIR=/mnt/HDD4/zlei0805/honor_degree/.tmp
export PIP_CACHE_DIR=/mnt/HDD4/zlei0805/honor_degree/.pip_cache
export CONDA_PKGS_DIRS=/mnt/HDD4/zlei0805/honor_degree/.conda_pkgs
export XDG_CACHE_HOME=/mnt/HDD4/zlei0805/honor_degree/.xdg_cache
export HF_HOME=/mnt/HDD4/zlei0805/honor_degree/.hf_home
export nnUNet_compile=false OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p "$RUN_BASE" "$(dirname "$READY")" "$TMPDIR" "$PIP_CACHE_DIR" "$CONDA_PKGS_DIRS" "$XDG_CACHE_HOME" "$HF_HOME"

"$PY" "$SCRIPT" stage --run-root "$RUN_ROOT" --ready "$READY" \
  --source-manifest "$SOURCE_MANIFEST" --splits-final "$SPLITS_FINAL" \
  --model-root "$MODEL_ROOT"

mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/state"
DONE=""
trap 'if [[ -z "$DONE" ]]; then printf "{\"status\":\"FAIL\",\"source_m0_lineage\":\"M0_V6_FIVEFOLD_OOF\"}\n" > "$RUN_ROOT/state/oof.fail"; fi' EXIT
printf '{"status":"RUNNING","source_m0_lineage":"M0_V6_FIVEFOLD_OOF","partitions":["train","val"]}\n' > "$RUN_ROOT/state/oof.running"

run_queue() {
  local gpu="$1"; shift
  for fold in "$@"; do
    CUDA_VISIBLE_DEVICES="$gpu" "$PY" "$SCRIPT" run-fold \
      --run-root "$RUN_ROOT" --fold "$fold" --device cuda:0 \
      > "$RUN_ROOT/logs/fold_${fold}.log" 2>&1
  done
}
set +e
if [[ "$GPU1" == "-1" ]]; then
  run_queue "$GPU0" 0 1 2 3 4
  RC0=$?
  RC1=0
else
  run_queue "$GPU0" 0 2 4 & PID0=$!
  run_queue "$GPU1" 1 3 & PID1=$!
  wait "$PID0"; RC0=$?
  wait "$PID1"; RC1=$?
fi
set -e
if [[ "$RC0" -ne 0 || "$RC1" -ne 0 ]]; then
  echo "M0-v6 OOF queue failed: gpu0=$RC0 gpu1=$RC1" >&2
  exit 1
fi

"$PY" "$SCRIPT" finalize --run-root "$RUN_ROOT" --ready "$READY" \
  > "$RUN_ROOT/logs/finalize.log" 2>&1
printf '{"status":"PASS","source_m0_lineage":"M0_V6_FIVEFOLD_OOF","ready":"%s"}\n' "$READY" > "$RUN_ROOT/state/oof.done"
rm -f "$RUN_ROOT/state/oof.running"
DONE=1
