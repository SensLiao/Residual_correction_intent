#!/usr/bin/env bash
# B1 same-editor gold-call ceiling pass for the R13 effect-val arms.
#
# The four effect-val editors (J6/J7/J8/J9) were evaluated against predicted
# compiler calls; their eval JSONs leave editor_gold_call_ceiling and
# predicted_vs_gold_same_editor_gap null.  This pass closes that gap without
# touching any frozen artifact: it builds evaluator-lane oracle calls from the
# VAL labels, re-runs each FROZEN editor checkpoint against those gold calls,
# and evaluates the predicted vs gold manifests together so the oracle pass
# emits the ceiling and the paired same-editor gap.
#
# Same-checkpoint binding: the predicted editor receipt and the gold editor
# receipt must hash to the identical editor checkpoint (enforced by the
# evaluator), so this pass only reuses $EFFECT_ROOT/models/$arm.pt.
set -euo pipefail

if [[ $# -lt 5 || $# -gt 6 ]]; then
  echo "usage: $0 <r13-main-run-root> <effect-val-run-root> <fresh-gold-run-root> <gpu0> <gpu1> [--dry-run]" >&2
  exit 2
fi
DRY_RUN=0
if [[ $# -eq 6 ]]; then
  [[ "$6" == "--dry-run" ]] || exit 2
  DRY_RUN=1
fi
PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
RUN_BASE="$PROJECT_ROOT/route_a/runs"
R13_ROOT=$(readlink -m "$1"); EFFECT_ROOT=$(readlink -m "$2"); ROOT=$(readlink -m "$3")
GPU0="$4"; GPU1="$5"
case "$R13_ROOT" in "$RUN_BASE"/PETCT-R13-MAIN-*) ;; *) exit 2;; esac
case "$EFFECT_ROOT" in "$RUN_BASE"/PETCT-R13-EFFECT-VAL-*) ;; *) exit 2;; esac
case "$ROOT" in "$RUN_BASE"/PETCT-R13-GOLD-CEILING-*) ;; *) exit 2;; esac
[[ "$GPU0" != "$GPU1" && "$GPU0" =~ ^[0-9]+$ && "$GPU1" =~ ^[0-9]+$ ]] || exit 2

DATA="$R13_ROOT/R13-main"; LINEAGE="$DATA/lineage-receipt.json"
INFERENCE="$DATA/inference-visible/episodes.jsonl"; LABELS="$DATA/label-only/labels.jsonl"
AUDIT="$DATA/audit-only/episodes.jsonl"
CANDIDATES="$DATA/inference-visible/candidates"; POINTERS="$DATA/label-only/pointer-targets"
PY="$PROJECT_ROOT/envs/petct_nnunet_v281/bin/python"

# Immutable effect-val inputs: predicted compiler calls plus the frozen editor
# checkpoints and their predicted-call manifests/receipts.
CALLS="$EFFECT_ROOT/predictions/J9C-val.jsonl"
CALL_RECEIPT="$EFFECT_ROOT/predictions/J9C-val.receipt.json"
COMPILER="$EFFECT_ROOT/models/J9C.pt"
ORACLE_CALLS="$ROOT/predictions/oracle-calls-val.jsonl"
ORACLE_RECEIPT="$ROOT/predictions/oracle-calls-val.receipt.json"

# Command builders print one argument per line; the dry run renders them as a
# JSON plan and the real run executes them verbatim, so the two cannot drift.
cmd_oracle_calls() {
  # R13-main natural labels carry no matched_state_group_id; the frozen v3
  # builder requires matched triplets, so the natural-lane variant is the only
  # legal producer here (same derive loop, require_matched_groups=False).
  printf '%s\n' \
    "$PY" "$PROJECT_ROOT/scripts/evaluation/render_petct_gold_program_calls_natural.py" \
    "--labels" "$LABELS" "--candidates" "$CANDIDATES" "--pointer-targets" "$POINTERS" \
    "--partition" "val" "--output" "$ORACLE_CALLS" "--receipt" "$ORACLE_RECEIPT"
}
cmd_arm_infer() {
  local arm="$1" gpu="$2"
  printf '%s\n' "env" "CUDA_VISIBLE_DEVICES=$gpu" "$PY" \
    "$PROJECT_ROOT/scripts/editor/infer_petct_program_editor_v3.py" \
    "--episodes" "$INFERENCE" "--candidates" "$CANDIDATES" \
    "--program-predictions" "$ORACLE_CALLS" "--program-receipt" "$ORACLE_RECEIPT" \
    "--editor-checkpoint" "$EFFECT_ROOT/models/$arm.pt" \
    "--lineage-receipt" "$LINEAGE"
  [[ "$arm" != "J8" ]] || printf '%s\n' "--compiler-checkpoint" "$COMPILER"
  printf '%s\n' \
    "--partition" "val" \
    "--output-dir" "$ROOT/predictions/$arm-gold-deltas" \
    "--output-manifest" "$ROOT/predictions/$arm-gold-val.jsonl" \
    "--receipt" "$ROOT/predictions/$arm-gold-val.receipt.json" \
    "--device" "cuda"
}
cmd_arm_evaluate() {
  local arm="$1"
  printf '%s\n' "$PY" \
    "$PROJECT_ROOT/scripts/evaluation/evaluate_petct_program_v3.py" \
    "--predictions" "$CALLS" "--prediction-receipt" "$CALL_RECEIPT" \
    "--lineage-receipt" "$LINEAGE" "--labels" "$LABELS" \
    "--inference-manifest" "$INFERENCE" "--candidates" "$CANDIDATES" \
    "--pointer-targets" "$POINTERS" "--audit-manifest" "$AUDIT" \
    "--editor-predictions" "$EFFECT_ROOT/predictions/$arm-val.jsonl" \
    "--editor-receipt" "$EFFECT_ROOT/predictions/$arm-val.receipt.json" \
    "--oracle-editor-predictions" "$ROOT/predictions/$arm-gold-val.jsonl" \
    "--oracle-editor-receipt" "$ROOT/predictions/$arm-gold-val.receipt.json" \
    "--partition" "val" "--output" "$ROOT/evaluation/$arm-gold.json"
}
if [[ "$DRY_RUN" -eq 1 ]]; then
  emit_plan() {
    local name="$1"; shift
    local args=()
    while IFS= read -r line; do [[ -z "$line" ]] || args+=("$line"); done < <("$@")
    printf '{"step":"%s","commands":[' "$name"
    local separator=""
    local arg
    for arg in "${args[@]}"; do
      printf '%s"%s"' "$separator" "$(printf '%s' "$arg" | sed 's/\\/\\\\/g; s/"/\\"/g')"
      separator=","
    done
    printf ']}\n'
  }
  emit_plan oracle-calls cmd_oracle_calls
  emit_plan J9-gold-infer cmd_arm_infer J9 "$GPU0"
  emit_plan J9-gold-evaluate cmd_arm_evaluate J9
  emit_plan J8-gold-infer cmd_arm_infer J8 "$GPU0"
  emit_plan J8-gold-evaluate cmd_arm_evaluate J8
  emit_plan J6-gold-infer cmd_arm_infer J6 "$GPU1"
  emit_plan J6-gold-evaluate cmd_arm_evaluate J6
  emit_plan J7-gold-infer cmd_arm_infer J7 "$GPU1"
  emit_plan J7-gold-evaluate cmd_arm_evaluate J7
  printf '{"step":"arm-semantics","J6":"program-blind spatial-only editor; the gold-call run removes abstention zeroing only","J7":"flat gold family action","J8":"continuous state readout plus gold call","J9":"typed gold family+operand call","shared":"same frozen editor checkpoint as the predicted run; oracle calls stay in the evaluator lane","partition":"val"}\n'
  exit 0
fi

for path in "$LINEAGE" "$INFERENCE" "$LABELS" "$AUDIT" "$CALLS" "$CALL_RECEIPT" "$COMPILER"; do
  [[ -f "$path" && ! -L "$path" ]] || exit 2
done
[[ -d "$CANDIDATES" && -d "$POINTERS" ]] || exit 2
[[ -f "$EFFECT_ROOT/state/effect_val.done" ]] || exit 2
for arm in J6 J7 J8 J9; do
  for path in "$EFFECT_ROOT/models/$arm.pt" "$EFFECT_ROOT/predictions/$arm-val.jsonl" "$EFFECT_ROOT/predictions/$arm-val.receipt.json"; do
    [[ -f "$path" && ! -L "$path" ]] || exit 2
  done
done
[[ ! -e "$ROOT" && ! -L "$ROOT" ]] || exit 2
compute=$(nvidia-smi --id="$GPU0,$GPU1" --query-compute-apps=gpu_uuid --format=csv,noheader 2>/dev/null | sed '/^[[:space:]]*$/d' || true)
[[ -z "$compute" ]] || exit 75

export TMPDIR=/mnt/HDD4/zlei0805/honor_degree/.tmp
export PIP_CACHE_DIR=/mnt/HDD4/zlei0805/honor_degree/.pip_cache
export CONDA_PKGS_DIRS=/mnt/HDD4/zlei0805/honor_degree/.conda_pkgs
export XDG_CACHE_HOME=/mnt/HDD4/zlei0805/honor_degree/.xdg_cache
export HF_HOME=/mnt/HDD4/zlei0805/honor_degree/.hf_home
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR" "$CONDA_PKGS_DIRS" "$XDG_CACHE_HOME" "$HF_HOME"
mkdir -p "$ROOT/logs" "$ROOT/state" "$ROOT/predictions" "$ROOT/evaluation"
DONE=""
trap 'if [[ -z "$DONE" ]]; then printf "{\"status\":\"FAIL\",\"partition\":\"val\",\"source_m0_lineage\":\"M0_V6_FIVEFOLD_OOF\"}\n" > "$ROOT/state/gold_call_ceiling.fail"; fi' EXIT
printf '{"status":"RUNNING","partition":"val","source_m0_lineage":"M0_V6_FIVEFOLD_OOF"}\n' > "$ROOT/state/gold_call_ceiling.running"

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
run_arm() {
  local arm="$1" gpu="$2"
  local CMD=()
  while IFS= read -r line; do [[ -z "$line" ]] || CMD+=("$line"); done < <(cmd_arm_infer "$arm" "$gpu")
  run_step "$arm-gold-infer" "${CMD[@]}"
  CMD=()
  while IFS= read -r line; do [[ -z "$line" ]] || CMD+=("$line"); done < <(cmd_arm_evaluate "$arm")
  run_step "$arm-gold-evaluate" "${CMD[@]}"
}
gold_queue() {
  local gpu="$1"; shift
  local arm
  for arm in "$@"; do run_arm "$arm" "$gpu"; done
}

CMD_ORACLE=()
while IFS= read -r line; do [[ -z "$line" ]] || CMD_ORACLE+=("$line"); done < <(cmd_oracle_calls)
run_step oracle-calls "${CMD_ORACLE[@]}"
gold_queue "$GPU0" J9 J8 & P0=$!
gold_queue "$GPU1" J6 J7 & P1=$!
set +e; wait "$P0"; R0=$?; wait "$P1"; R1=$?; set -e
[[ "$R0" -eq 0 && "$R1" -eq 0 ]] || exit 1

printf '{"status":"PASS","partition":"val","arms":["J6","J7","J8","J9"],"arm_semantics":{"J6":"program-blind ceiling; abstention-removal only"},"source_m0_lineage":"M0_V6_FIVEFOLD_OOF"}\n' > "$ROOT/state/gold_call_ceiling.done"
rm -f "$ROOT/state/gold_call_ceiling.running"
DONE=1
