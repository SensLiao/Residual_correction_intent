#!/usr/bin/env bash
# B2 five-round 2D trajectory ceiling pass for the R13 effect-val arms.
#
# Round 1 reuses the natural-lane oracle calls built by the B1 gold-call
# ceiling pass; rounds 2-5 regenerate official-simulator scribbles on the
# current-state residual and derive residual-driven gold calls.  Every round
# advances a 3D state volume with the ACTUAL frozen editor output restricted
# to the prompted plane (plane-edit reconstruction); per-round delta Dice is
# recorded in the 2d_prompted_plane domain (crop-space + state-grid).
#
# Immutable inputs: R13-main lanes + materialized source case manifest,
# effect-val editor/compiler checkpoints, B1 gold-ceiling oracle calls.
# All new artifacts land in the fresh PETCT-R13-B2-TRAJECTORY-* run root.
set -euo pipefail

if [[ $# -lt 7 || $# -gt 8 ]]; then
  echo "usage: $0 <r13-main-run-root> <effect-val-run-root> <gold-ceiling-run-root> <fresh-b2-run-root> <case-manifest-file> <gpu0> <gpu1> [--dry-run]" >&2
  exit 2
fi
DRY_RUN=0
if [[ $# -eq 8 ]]; then
  [[ "$8" == "--dry-run" ]] || exit 2
  DRY_RUN=1
fi
PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
RUN_BASE="$PROJECT_ROOT/route_a/runs"
R13_ROOT=$(readlink -m "$1"); EFFECT_ROOT=$(readlink -m "$2")
GOLD_ROOT=$(readlink -m "$3"); ROOT=$(readlink -m "$4")
CASE_MANIFEST=$(readlink -m "$5"); GPU0="$6"; GPU1="$7"
case "$R13_ROOT" in "$RUN_BASE"/PETCT-R13-MAIN-*) ;; *) exit 2;; esac
case "$EFFECT_ROOT" in "$RUN_BASE"/PETCT-R13-EFFECT-VAL-*) ;; *) exit 2;; esac
case "$GOLD_ROOT" in "$RUN_BASE"/PETCT-R13-GOLD-CEILING-*) ;; *) exit 2;; esac
case "$ROOT" in "$RUN_BASE"/PETCT-R13-B2-TRAJECTORY-*) ;; *) exit 2;; esac
[[ "$GPU0" != "$GPU1" && "$GPU0" =~ ^[0-9]+$ && "$GPU1" =~ ^[0-9]+$ ]] || exit 2

DATA="$R13_ROOT/R13-main"; LINEAGE="$DATA/lineage-receipt.json"
EPISODES="$DATA/inference-visible/episodes.jsonl"
RICH="$DATA/audit-only/tensors-rich.jsonl"
CANDIDATES="$DATA/inference-visible/candidates"
PY="$PROJECT_ROOT/envs/petct_nnunet_v281/bin/python"

# Immutable upstream artifacts.
ORACLE_CALLS="$GOLD_ROOT/predictions/oracle-calls-val.jsonl"
ORACLE_RECEIPT="$GOLD_ROOT/predictions/oracle-calls-val.receipt.json"
COMPILER="$EFFECT_ROOT/models/J9C.pt"
SIMULATOR="$PROJECT_ROOT/upstream/autoPETV/interactive/simulate_scribbles.py"

# Command builders print one argument per line; the dry run renders them as a
# JSON plan and the real run executes them verbatim, so the two cannot drift.
cmd_arm_trajectory() {
  local arm="$1" gpu="$2"
  printf '%s\n' "env" "CUDA_VISIBLE_DEVICES=$gpu" "$PY" \
    "$PROJECT_ROOT/scripts/evaluation/run_petct_b2_trajectory_ceiling.py" \
    "--case-manifest" "$CASE_MANIFEST" \
    "--episodes" "$EPISODES" "--rich-manifest" "$RICH" \
    "--candidates" "$CANDIDATES" \
    "--oracle-calls" "$ORACLE_CALLS" "--oracle-receipt" "$ORACLE_RECEIPT" \
    "--editor-checkpoint" "$EFFECT_ROOT/models/$arm.pt" \
    "--lineage-receipt" "$LINEAGE"
  [[ "$arm" != "J8" ]] || printf '%s\n' "--compiler-checkpoint" "$COMPILER"
  printf '%s\n' \
    "--partition" "val" "--official-simulator" "$SIMULATOR" \
    "--device" "cuda" "--output" "$ROOT/evaluation/$arm-trajectory.json"
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
  emit_plan J9-trajectory cmd_arm_trajectory J9 "$GPU0"
  emit_plan J8-trajectory cmd_arm_trajectory J8 "$GPU0"
  emit_plan J6-trajectory cmd_arm_trajectory J6 "$GPU1"
  emit_plan J7-trajectory cmd_arm_trajectory J7 "$GPU1"
  printf '{"step":"arm-semantics","round_1":"natural_label_oracle (B1 gold-ceiling oracle calls)","rounds_2_to_5":"residual_driven_2d (DELETE_COMPONENT@cue-hit / COMPLETE_EXISTING@max-overlap else CREATE_NEW)","state_advance":"actual editor output on the prompted plane; plane-edit reconstruction; never GT substitution","domain":"2d_prompted_plane","J6":"program-blind editor; gold call still selects the REMOVE scope mask","checkpoint_lock":"effect-val frozen weights; no retraining","partition":"val"}\n'
  exit 0
fi

for path in "$LINEAGE" "$EPISODES" "$RICH" "$CASE_MANIFEST" "$ORACLE_CALLS" "$ORACLE_RECEIPT" "$COMPILER" "$SIMULATOR"; do
  [[ -f "$path" && ! -L "$path" ]] || exit 2
done
[[ -d "$CANDIDATES" ]] || exit 2
[[ -f "$EFFECT_ROOT/state/effect_val.done" ]] || exit 2
[[ -f "$GOLD_ROOT/state/gold_call_ceiling.done" ]] || exit 2
for arm in J6 J7 J8 J9; do
  [[ -f "$EFFECT_ROOT/models/$arm.pt" && ! -L "$EFFECT_ROOT/models/$arm.pt" ]] || exit 2
done
[[ ! -e "$ROOT" && ! -L "$ROOT" ]] || exit 2
compute=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader 2>/dev/null | sed '/^[[:space:]]*$/d' || true)
[[ -z "$compute" ]] || exit 75

export TMPDIR=/mnt/HDD4/zlei0805/honor_degree/.tmp
export PIP_CACHE_DIR=/mnt/HDD4/zlei0805/honor_degree/.pip_cache
export CONDA_PKGS_DIRS=/mnt/HDD4/zlei0805/honor_degree/.conda_pkgs
export XDG_CACHE_HOME=/mnt/HDD4/zlei0805/honor_degree/.xdg_cache
export HF_HOME=/mnt/HDD4/zlei0805/honor_degree/.hf_home
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR" "$CONDA_PKGS_DIRS" "$XDG_CACHE_HOME" "$HF_HOME"
mkdir -p "$ROOT/logs" "$ROOT/state" "$ROOT/evaluation"
DONE=""
trap 'if [[ -z "$DONE" ]]; then printf "{\"status\":\"FAIL\",\"partition\":\"val\",\"source_m0_lineage\":\"M0_V6_FIVEFOLD_OOF\"}\n" > "$ROOT/state/b2_trajectory.fail"; fi' EXIT
printf '{"status":"RUNNING","partition":"val","source_m0_lineage":"M0_V6_FIVEFOLD_OOF"}\n' > "$ROOT/state/b2_trajectory.running"

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
  while IFS= read -r line; do [[ -z "$line" ]] || CMD+=("$line"); done < <(cmd_arm_trajectory "$arm" "$gpu")
  run_step "$arm-trajectory" "${CMD[@]}"
}
trajectory_queue() {
  local gpu="$1"; shift
  local arm
  for arm in "$@"; do run_arm "$arm" "$gpu"; done
}

trajectory_queue "$GPU0" J9 J8 & P0=$!
trajectory_queue "$GPU1" J6 J7 & P1=$!
set +e; wait "$P0"; R0=$?; wait "$P1"; R1=$?; set -e
[[ "$R0" -eq 0 && "$R1" -eq 0 ]] || exit 1

printf '{"status":"PASS","partition":"val","arms":["J6","J7","J8","J9"],"gold_call_policy":{"round_1":"natural_label_oracle","rounds_2_to_5":"residual_driven_2d"},"source_m0_lineage":"M0_V6_FIVEFOLD_OOF"}\n' > "$ROOT/state/b2_trajectory.done"
rm -f "$ROOT/state/b2_trajectory.running"
DONE=1
