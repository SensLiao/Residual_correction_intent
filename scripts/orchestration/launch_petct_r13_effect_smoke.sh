#!/usr/bin/env bash
# One-epoch TRAIN/VAL smoke across J1/J2/J6/J7/J8/J9 and the J9C compiler.
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "usage: $0 <r13-main-run-root> <fresh-smoke-run-root> <gpu0> <gpu1>" >&2
  exit 2
fi
PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
RUN_BASE="$PROJECT_ROOT/route_a/runs"
R13_ROOT=$(readlink -m "$1")
ROOT=$(readlink -m "$2")
GPU0="$3"; GPU1="$4"
case "$R13_ROOT" in "$RUN_BASE"/PETCT-R13-MAIN-*) ;; *) echo "invalid R13 root" >&2; exit 2;; esac
case "$ROOT" in "$RUN_BASE"/PETCT-R13-SMOKE-*) ;; *) echo "invalid smoke root" >&2; exit 2;; esac
[[ "$GPU0" != "$GPU1" && "$GPU0" =~ ^[0-9]+$ && "$GPU1" =~ ^[0-9]+$ ]] || exit 2
[[ ! -e "$ROOT" && ! -L "$ROOT" ]] || { echo "smoke root exists" >&2; exit 2; }

DATA="$R13_ROOT/R13-main"
READY="$DATA/data-ready.json"
LINEAGE="$DATA/lineage-receipt.json"
MANIFEST_RECEIPT="$DATA/program-manifest-receipt.json"
INFERENCE="$DATA/inference-visible/episodes.jsonl"
LABELS="$DATA/label-only/labels.jsonl"
AUDIT="$DATA/audit-only/episodes.jsonl"
RICH="$DATA/audit-only/tensors-rich.jsonl"
CANDIDATES="$DATA/inference-visible/candidates"
POINTERS="$DATA/label-only/pointer-targets"
PY="$PROJECT_ROOT/envs/petct_nnunet_v281/bin/python"
LINEAGE_TOOL="$PROJECT_ROOT/scripts/common/petct_mainline_lineage.py"
for path in "$READY" "$LINEAGE" "$MANIFEST_RECEIPT" "$INFERENCE" "$LABELS" "$AUDIT" "$RICH"; do
  [[ -f "$path" && ! -L "$path" ]] || { echo "missing R13 input: $path" >&2; exit 2; }
done
[[ -d "$CANDIDATES" && -d "$POINTERS" ]] || { echo "missing component sidecar" >&2; exit 2; }

readarray -t BOUND < <("$PY" - "$LINEAGE" <<'PY'
import json,sys
d=json.load(open(sys.argv[1],encoding='utf-8'))
print(d['learning_split']['path'])
print(d['experiment_config']['path'])
PY
)
SPLIT="${BOUND[0]}"; CONFIG="${BOUND[1]}"
"$PY" "$LINEAGE_TOOL" validate-data --receipt "$READY" >/dev/null

compute=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader 2>/dev/null | sed '/^[[:space:]]*$/d' || true)
[[ -z "$compute" ]] || { echo "smoke requires exclusive idle GPUs" >&2; exit 75; }
export TMPDIR=/mnt/HDD4/zlei0805/honor_degree/.tmp
export PIP_CACHE_DIR=/mnt/HDD4/zlei0805/honor_degree/.pip_cache
export CONDA_PKGS_DIRS=/mnt/HDD4/zlei0805/honor_degree/.conda_pkgs
export XDG_CACHE_HOME=/mnt/HDD4/zlei0805/honor_degree/.xdg_cache
export HF_HOME=/mnt/HDD4/zlei0805/honor_degree/.hf_home
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR" "$CONDA_PKGS_DIRS" "$XDG_CACHE_HOME" "$HF_HOME"
mkdir -p "$ROOT/logs" "$ROOT/state" "$ROOT/models" "$ROOT/predictions" "$ROOT/evaluation"
printf '{"status":"PASS","gate":"0-A","data_ready":"%s"}\n' "$READY" > "$ROOT/state/gate0a.done"

DONE=""; STAGE="j9c_train"
trap 'if [[ -z "$DONE" ]]; then printf "{\"status\":\"FAIL\",\"stage\":\"%s\",\"source_m0_lineage\":\"M0_V6_FIVEFOLD_OOF\"}\n" "$STAGE" > "$ROOT/state/smoke.fail"; rm -f "$ROOT/state/smoke.running"; fi' EXIT
printf '{"status":"RUNNING","stage":"%s","partition":"val","source_m0_lineage":"M0_V6_FIVEFOLD_OOF"}\n' "$STAGE" > "$ROOT/state/smoke.running"

J9C="$ROOT/models/J9C.pt"
CUDA_VISIBLE_DEVICES="$GPU0" "$PY" "$PROJECT_ROOT/scripts/p2t/train_petct_program_v3.py" \
  --episodes "$INFERENCE" --labels "$LABELS" --learning-split "$SPLIT" \
  --manifest-receipt "$MANIFEST_RECEIPT" --lineage-receipt "$LINEAGE" \
  --candidates "$CANDIDATES" --pointer-targets "$POINTERS" \
  --experiment-config "$CONFIG" --output "$J9C" --arm J9C \
  --dataset-mode natural --epochs 1 --batch-size 16 --seed 3407 --device cuda \
  > "$ROOT/logs/J9C-train.log" 2>&1

STAGE="j9c_infer"
for i in 1 2 3; do
  CUDA_VISIBLE_DEVICES="$GPU0" "$PY" "$PROJECT_ROOT/scripts/p2t/infer_petct_program_v3.py" \
    --episodes "$INFERENCE" --candidates "$CANDIDATES" --checkpoint "$J9C" \
    --lineage-receipt "$LINEAGE" --partition val \
    --output "$ROOT/predictions/J9C-val-run$i.jsonl" \
    --receipt "$ROOT/predictions/J9C-val-run$i.receipt.json" \
    --batch-size 16 --device cuda > "$ROOT/logs/J9C-infer-run$i.log" 2>&1
done
cmp -s "$ROOT/predictions/J9C-val-run1.jsonl" "$ROOT/predictions/J9C-val-run2.jsonl"
cmp -s "$ROOT/predictions/J9C-val-run1.jsonl" "$ROOT/predictions/J9C-val-run3.jsonl"
CALLS="$ROOT/predictions/J9C-val-run1.jsonl"
CALL_RECEIPT="$ROOT/predictions/J9C-val-run1.receipt.json"
printf '{"status":"PASS","gate":"0-B","determinism_3x":true}\n' > "$ROOT/state/gate0b.done"

for ARM in J6 J7 J9; do
  STAGE="${ARM}_train"
  CUDA_VISIBLE_DEVICES="$GPU1" "$PY" "$PROJECT_ROOT/scripts/editor/train_petct_program_editor_v3.py" \
    --episodes "$INFERENCE" --labels "$LABELS" --learning-split "$SPLIT" \
    --manifest-receipt "$MANIFEST_RECEIPT" --lineage-receipt "$LINEAGE" \
    --candidates "$CANDIDATES" --pointer-targets "$POINTERS" \
    --experiment-config "$CONFIG" --output "$ROOT/models/${ARM}.pt" --arm "$ARM" \
    --call-source gold --dataset-mode natural --epochs 1 --batch-size 8 \
    --seed 3407 --device cuda > "$ROOT/logs/${ARM}-train.log" 2>&1
done
STAGE="J8_train"
CUDA_VISIBLE_DEVICES="$GPU1" "$PY" "$PROJECT_ROOT/scripts/editor/train_petct_program_editor_v3.py" \
  --episodes "$INFERENCE" --labels "$LABELS" --learning-split "$SPLIT" \
  --manifest-receipt "$MANIFEST_RECEIPT" --lineage-receipt "$LINEAGE" \
  --candidates "$CANDIDATES" --pointer-targets "$POINTERS" --compiler-checkpoint "$J9C" \
  --experiment-config "$CONFIG" --output "$ROOT/models/J8.pt" --arm J8 \
  --call-source gold --dataset-mode natural --epochs 1 --batch-size 8 \
  --seed 3407 --device cuda > "$ROOT/logs/J8-train.log" 2>&1

for ARM in J6 J7 J8 J9; do
  STAGE="${ARM}_infer"
  EXTRA=()
  [[ "$ARM" != "J8" ]] || EXTRA=(--compiler-checkpoint "$J9C")
  CUDA_VISIBLE_DEVICES="$GPU1" "$PY" "$PROJECT_ROOT/scripts/editor/infer_petct_program_editor_v3.py" \
    --episodes "$INFERENCE" --candidates "$CANDIDATES" \
    --program-predictions "$CALLS" --program-receipt "$CALL_RECEIPT" \
    --editor-checkpoint "$ROOT/models/${ARM}.pt" --lineage-receipt "$LINEAGE" \
    "${EXTRA[@]}" --partition val --output-dir "$ROOT/predictions/${ARM}-deltas" \
    --output-manifest "$ROOT/predictions/${ARM}-val.jsonl" \
    --receipt "$ROOT/predictions/${ARM}-val.receipt.json" --batch-size 16 --device cuda \
    > "$ROOT/logs/${ARM}-infer.log" 2>&1
  STAGE="${ARM}_evaluate"
  "$PY" "$PROJECT_ROOT/scripts/evaluation/evaluate_petct_program_v3.py" \
    --predictions "$CALLS" --prediction-receipt "$CALL_RECEIPT" \
    --lineage-receipt "$LINEAGE" --labels "$LABELS" --inference-manifest "$INFERENCE" \
    --candidates "$CANDIDATES" --pointer-targets "$POINTERS" --audit-manifest "$AUDIT" \
    --editor-predictions "$ROOT/predictions/${ARM}-val.jsonl" \
    --editor-receipt "$ROOT/predictions/${ARM}-val.receipt.json" \
    --partition val --bootstrap-samples 50 --output "$ROOT/evaluation/${ARM}.json" \
    > "$ROOT/logs/${ARM}-evaluate.log" 2>&1
done

for ARM in J1 J2; do
  STAGE="${ARM}_train"
  GPU="$GPU0"; [[ "$ARM" != "J2" ]] || GPU="$GPU1"
  CUDA_VISIBLE_DEVICES="$GPU" "$PY" "$PROJECT_ROOT/scripts/p2t/train_petct_p2t.py" \
    --manifest "$RICH" --learning-split "$SPLIT" --experiment-config "$CONFIG" \
    --r13-data-ready "$READY" --baseline-arm "$ARM" --output "$ROOT/models/${ARM}.pt" \
    --smoke-one-epoch --epochs 1 --seed 3407 --device cuda > "$ROOT/logs/${ARM}-train.log" 2>&1
  STAGE="${ARM}_evaluate"
  CUDA_VISIBLE_DEVICES="$GPU" "$PY" "$PROJECT_ROOT/scripts/evaluation/evaluate_petct_p2t.py" \
    --manifest "$RICH" --training-manifest "$RICH" --experiment-config "$CONFIG" \
    --learning-split "$SPLIT" --checkpoint "$ROOT/models/${ARM}.pt" \
    --r13-data-ready "$READY" --partition val \
    --predictions "$ROOT/predictions/${ARM}-val.jsonl" \
    --paired-evaluation-rows "$ROOT/evaluation/${ARM}-paired.jsonl" \
    --metrics "$ROOT/evaluation/${ARM}.json" --device cuda \
    > "$ROOT/logs/${ARM}-evaluate.log" 2>&1
done

test -s "$ROOT/evaluation/J9.json"
test -s "$ROOT/predictions/J9-val.receipt.json"
printf '{"status":"PASS","gate":"0-C","condition":"R13 predicted-program natural VAL editor diagnostic"}\n' > "$ROOT/state/gate0c.done"

printf '{"status":"PASS","partition":"val","arms":["J1","J2","J6","J7","J8","J9","J9C"],"source_m0_lineage":"M0_V6_FIVEFOLD_OOF"}\n' > "$ROOT/state/smoke.done"
rm -f "$ROOT/state/smoke.running"
DONE=1
