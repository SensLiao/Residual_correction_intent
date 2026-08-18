#!/usr/bin/env bash
# Full frozen-budget TRAIN/VAL effect wave for the R13 mainline.
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "usage: $0 <r13-main-run-root> <fresh-val-run-root> <gpu0> <gpu1>" >&2
  exit 2
fi
PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
RUN_BASE="$PROJECT_ROOT/route_a/runs"
R13_ROOT=$(readlink -m "$1"); ROOT=$(readlink -m "$2"); GPU0="$3"; GPU1="$4"
case "$R13_ROOT" in "$RUN_BASE"/PETCT-R13-MAIN-*) ;; *) exit 2;; esac
case "$ROOT" in "$RUN_BASE"/PETCT-R13-EFFECT-VAL-*) ;; *) exit 2;; esac
[[ "$GPU0" != "$GPU1" && "$GPU0" =~ ^[0-9]+$ && "$GPU1" =~ ^[0-9]+$ ]] || exit 2
[[ ! -e "$ROOT" && ! -L "$ROOT" ]] || exit 2

DATA="$R13_ROOT/R13-main"; READY="$DATA/data-ready.json"; LINEAGE="$DATA/lineage-receipt.json"
MANIFEST_RECEIPT="$DATA/program-manifest-receipt.json"
INFERENCE="$DATA/inference-visible/episodes.jsonl"; LABELS="$DATA/label-only/labels.jsonl"
AUDIT="$DATA/audit-only/episodes.jsonl"; RICH="$DATA/audit-only/tensors-rich.jsonl"
CANDIDATES="$DATA/inference-visible/candidates"; POINTERS="$DATA/label-only/pointer-targets"
PY="$PROJECT_ROOT/envs/petct_nnunet_v281/bin/python"
LINEAGE_TOOL="$PROJECT_ROOT/scripts/common/petct_mainline_lineage.py"
for path in "$READY" "$LINEAGE" "$MANIFEST_RECEIPT" "$INFERENCE" "$LABELS" "$AUDIT" "$RICH"; do
  [[ -f "$path" && ! -L "$path" ]] || exit 2
done
[[ -d "$CANDIDATES" && -d "$POINTERS" ]] || exit 2
readarray -t BOUND < <("$PY" - "$LINEAGE" <<'PY'
import json,sys
d=json.load(open(sys.argv[1],encoding='utf-8'))
print(d['learning_split']['path']); print(d['experiment_config']['path'])
PY
)
SPLIT="${BOUND[0]}"; CONFIG="${BOUND[1]}"
"$PY" "$LINEAGE_TOOL" validate-data --receipt "$READY" >/dev/null
compute=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader 2>/dev/null | sed '/^[[:space:]]*$/d' || true)
[[ -z "$compute" ]] || exit 75

export TMPDIR=/mnt/HDD4/workspace/honor_degree/.tmp
export PIP_CACHE_DIR=/mnt/HDD4/workspace/honor_degree/.pip_cache
export CONDA_PKGS_DIRS=/mnt/HDD4/workspace/honor_degree/.conda_pkgs
export XDG_CACHE_HOME=/mnt/HDD4/workspace/honor_degree/.xdg_cache
export HF_HOME=/mnt/HDD4/workspace/honor_degree/.hf_home
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR" "$CONDA_PKGS_DIRS" "$XDG_CACHE_HOME" "$HF_HOME"
mkdir -p "$ROOT/logs" "$ROOT/state" "$ROOT/models" "$ROOT/predictions" "$ROOT/evaluation"
DONE=""
trap 'if [[ -z "$DONE" ]]; then printf "{\"status\":\"FAIL\",\"source_m0_lineage\":\"M0_V6_FIVEFOLD_OOF\"}\n" > "$ROOT/state/effect_val.fail"; fi' EXIT
printf '{"status":"RUNNING","partition":"val","source_m0_lineage":"M0_V6_FIVEFOLD_OOF"}\n' > "$ROOT/state/effect_val.running"

run_step() {
  local name="$1"; shift
  printf '{"status":"RUNNING","stage":"%s"}\n' "$name" > "$ROOT/state/$name.running"
  if "$@" > "$ROOT/logs/$name.log" 2>&1; then
    mv "$ROOT/state/$name.running" "$ROOT/state/$name.done"
  else
    mv "$ROOT/state/$name.running" "$ROOT/state/$name.fail"
    return 1
  fi
}

train_j9c() {
  run_step J9C-train env CUDA_VISIBLE_DEVICES="$GPU0" "$PY" "$PROJECT_ROOT/scripts/p2t/train_petct_program_v3.py" \
    --episodes "$INFERENCE" --labels "$LABELS" --learning-split "$SPLIT" \
    --manifest-receipt "$MANIFEST_RECEIPT" --lineage-receipt "$LINEAGE" \
    --candidates "$CANDIDATES" --pointer-targets "$POINTERS" \
    --experiment-config "$CONFIG" --output "$ROOT/models/J9C.pt" --arm J9C \
    --dataset-mode natural --seed 3407 --device cuda
}
baseline_queue() {
  local arm
  for arm in J1 J2; do
    run_step "$arm-train" env CUDA_VISIBLE_DEVICES="$GPU1" "$PY" "$PROJECT_ROOT/scripts/p2t/train_petct_p2t.py" \
      --manifest "$RICH" --learning-split "$SPLIT" --experiment-config "$CONFIG" \
      --r13-data-ready "$READY" --baseline-arm "$arm" --output "$ROOT/models/$arm.pt" \
      --seed 3407 --device cuda
    run_step "$arm-evaluate" env CUDA_VISIBLE_DEVICES="$GPU1" "$PY" "$PROJECT_ROOT/scripts/evaluation/evaluate_petct_p2t.py" \
      --manifest "$RICH" --training-manifest "$RICH" --experiment-config "$CONFIG" \
      --learning-split "$SPLIT" --checkpoint "$ROOT/models/$arm.pt" \
      --r13-data-ready "$READY" --partition val \
      --predictions "$ROOT/predictions/$arm-val.jsonl" \
      --paired-evaluation-rows "$ROOT/evaluation/$arm-paired.jsonl" \
      --metrics "$ROOT/evaluation/$arm.json" --device cuda
  done
}
train_j9c & P0=$!; baseline_queue & P1=$!
set +e; wait "$P0"; R0=$?; wait "$P1"; R1=$?; set -e
[[ "$R0" -eq 0 && "$R1" -eq 0 ]] || exit 1

CALLS="$ROOT/predictions/J9C-val.jsonl"; CALL_RECEIPT="$ROOT/predictions/J9C-val.receipt.json"
run_step J9C-infer env CUDA_VISIBLE_DEVICES="$GPU0" "$PY" "$PROJECT_ROOT/scripts/p2t/infer_petct_program_v3.py" \
  --episodes "$INFERENCE" --candidates "$CANDIDATES" --checkpoint "$ROOT/models/J9C.pt" \
  --lineage-receipt "$LINEAGE" --partition val --output "$CALLS" \
  --receipt "$CALL_RECEIPT" --device cuda

editor_queue() {
  local gpu="$1"; shift
  local arm extra
  for arm in "$@"; do
    extra=()
    [[ "$arm" != "J8" ]] || extra=(--compiler-checkpoint "$ROOT/models/J9C.pt")
    run_step "$arm-train" env CUDA_VISIBLE_DEVICES="$gpu" "$PY" "$PROJECT_ROOT/scripts/editor/train_petct_program_editor_v3.py" \
      --episodes "$INFERENCE" --labels "$LABELS" --learning-split "$SPLIT" \
      --manifest-receipt "$MANIFEST_RECEIPT" --lineage-receipt "$LINEAGE" \
      --candidates "$CANDIDATES" --pointer-targets "$POINTERS" "${extra[@]}" \
      --experiment-config "$CONFIG" --output "$ROOT/models/$arm.pt" --arm "$arm" \
      --call-source gold --dataset-mode natural --seed 3407 --device cuda
    run_step "$arm-infer" env CUDA_VISIBLE_DEVICES="$gpu" "$PY" "$PROJECT_ROOT/scripts/editor/infer_petct_program_editor_v3.py" \
      --episodes "$INFERENCE" --candidates "$CANDIDATES" --program-predictions "$CALLS" \
      --program-receipt "$CALL_RECEIPT" --editor-checkpoint "$ROOT/models/$arm.pt" \
      --lineage-receipt "$LINEAGE" "${extra[@]}" --partition val \
      --output-dir "$ROOT/predictions/$arm-deltas" \
      --output-manifest "$ROOT/predictions/$arm-val.jsonl" \
      --receipt "$ROOT/predictions/$arm-val.receipt.json" --device cuda
    run_step "$arm-evaluate" "$PY" "$PROJECT_ROOT/scripts/evaluation/evaluate_petct_program_v3.py" \
      --predictions "$CALLS" --prediction-receipt "$CALL_RECEIPT" --lineage-receipt "$LINEAGE" \
      --labels "$LABELS" --inference-manifest "$INFERENCE" --candidates "$CANDIDATES" \
      --pointer-targets "$POINTERS" --audit-manifest "$AUDIT" \
      --editor-predictions "$ROOT/predictions/$arm-val.jsonl" \
      --editor-receipt "$ROOT/predictions/$arm-val.receipt.json" \
      --partition val --output "$ROOT/evaluation/$arm.json"
  done
}
editor_queue "$GPU0" J9 J8 & P0=$!; editor_queue "$GPU1" J6 J7 & P1=$!
set +e; wait "$P0"; R0=$?; wait "$P1"; R1=$?; set -e
[[ "$R0" -eq 0 && "$R1" -eq 0 ]] || exit 1

printf '{"status":"PASS","partition":"val","arms":["J1","J2","J6","J7","J8","J9","J9C"],"source_m0_lineage":"M0_V6_FIVEFOLD_OOF"}\n' > "$ROOT/state/effect_val.done"
rm -f "$ROOT/state/effect_val.running"
DONE=1
