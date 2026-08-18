#!/usr/bin/env bash
# Build R13-trajectory-5r from the frozen R13-main residuals. TRAIN/VAL only,
# pure CPU.  The five-round corpus differs from R13-main only in round count:
# identical strategy geometry, sibling structure, exclusion rules and lane
# split, with teacher-forced oracle state progression across round_index 0..4.
set -euo pipefail

usage() {
  echo "usage: $0 --run-root DIR --r13-main-data DIR --r13-main-data-ready FILE --oof-ready FILE --learning-split FILE --experiment-config FILE --official-simulator FILE --official-runtime-manifest FILE [--dry-run]" >&2
}

DRY_RUN=0
RUN_ROOT=""; R13_MAIN_DATA=""; R13_MAIN_DATA_READY=""; OOF_READY=""; LEARNING_SPLIT=""
EXPERIMENT_CONFIG=""; OFFICIAL_SIMULATOR=""; OFFICIAL_RUNTIME_MANIFEST=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-root) RUN_ROOT="${2:-}"; shift 2;;
    --r13-main-data) R13_MAIN_DATA="${2:-}"; shift 2;;
    --r13-main-data-ready) R13_MAIN_DATA_READY="${2:-}"; shift 2;;
    --oof-ready) OOF_READY="${2:-}"; shift 2;;
    --learning-split) LEARNING_SPLIT="${2:-}"; shift 2;;
    --experiment-config) EXPERIMENT_CONFIG="${2:-}"; shift 2;;
    --official-simulator) OFFICIAL_SIMULATOR="${2:-}"; shift 2;;
    --official-runtime-manifest) OFFICIAL_RUNTIME_MANIFEST="${2:-}"; shift 2;;
    --dry-run) DRY_RUN=1; shift;;
    *) usage; exit 2;;
  esac
done
for value in "$RUN_ROOT" "$R13_MAIN_DATA" "$R13_MAIN_DATA_READY" "$OOF_READY" "$LEARNING_SPLIT" "$EXPERIMENT_CONFIG" "$OFFICIAL_SIMULATOR" "$OFFICIAL_RUNTIME_MANIFEST"; do
  [[ -n "$value" ]] || { usage; exit 2; }
done

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
RUN_BASE="$PROJECT_ROOT/route_a/runs"
RUN_ROOT=$(readlink -m "$RUN_ROOT")
case "$RUN_ROOT" in "$RUN_BASE"/PETCT-R13-TRAJECTORY-5R-*) ;; *) echo "invalid R13 trajectory run root" >&2; exit 2;; esac

PY="$PROJECT_ROOT/envs/petct_nnunet_v281/bin/python"
MAINLINE_TOOL="$PROJECT_ROOT/scripts/common/petct_mainline_lineage.py"
TRAJECTORY_TOOL="$PROJECT_ROOT/scripts/common/petct_trajectory_lineage.py"
# Frozen R13-main residuals are consumed, never regenerated; their hash
# binding to the active single-round corpus is the frozen-input precheck.
RESIDUALS="$R13_MAIN_DATA/audit-only/residuals.jsonl"
RESIDUAL_READY="$R13_MAIN_DATA/audit-only/RESIDUAL_READY.json"
DATA="$RUN_ROOT/R13-trajectory-5r"
VISIBLE="$DATA/inference-visible"
LABEL="$DATA/label-only"
AUDIT="$DATA/audit-only"
LINEAGE="$DATA/trajectory-lineage-receipt.json"
MANIFEST_RECEIPT="$DATA/trajectory-program-manifest-receipt.json"
DATA_READY="$DATA/trajectory-data-ready.json"
STATE="$RUN_ROOT/state/r13_trajectory_5r"

# Command builders print one argument per line; the dry run renders them as a
# JSON plan and the real run executes them verbatim, so the two cannot drift.
cmd_preflight() {
  printf '%s\n' "$PY" "$MAINLINE_TOOL" validate-data --receipt "$R13_MAIN_DATA_READY"
}
cmd_lineage() {
  printf '%s\n' "$PY" "$TRAJECTORY_TOOL" issue \
    --oof-ready "$OOF_READY" --learning-split "$LEARNING_SPLIT" \
    --experiment-config "$EXPERIMENT_CONFIG" --output "$LINEAGE"
}
cmd_trajectories() {
  printf '%s\n' "$PY" "$PROJECT_ROOT/scripts/data/build_petct_r13_trajectory_5r.py" \
    --residual-manifest "$RESIDUALS" --residual-ready "$RESIDUAL_READY" \
    --official-simulator "$OFFICIAL_SIMULATOR" \
    --official-runtime-manifest "$OFFICIAL_RUNTIME_MANIFEST" \
    --experiment-config "$EXPERIMENT_CONFIG" --learning-split "$LEARNING_SPLIT" \
    --partitions train val --strategy-mode primary --seed 42 --lane natural \
    --oof-ready "$OOF_READY" \
    --visible-root "$VISIBLE/episode-documents" \
    --evaluation-root "$AUDIT/episode-documents" \
    --authorized-root "$AUDIT/authorized-masks" \
    --state-root "$AUDIT/trajectory-states" \
    --output-manifest "$AUDIT/episodes-rich.jsonl" \
    --trajectories "$AUDIT/trajectories.jsonl" \
    --exclusions "$AUDIT/episode-exclusions.jsonl" \
    --ready-receipt "$AUDIT/TRAJECTORY_EPISODES_READY.json"
}
cmd_tensors() {
  printf '%s\n' "$PY" "$PROJECT_ROOT/scripts/data/materialize_petct_r13_trajectory_5r_tensors.py" \
    --episode-manifest "$AUDIT/episodes-rich.jsonl" \
    --visible-root "$VISIBLE/tensors" --evaluation-root "$AUDIT/tensors" \
    --output-manifest "$AUDIT/tensors-rich.jsonl" \
    --experiment-config "$EXPERIMENT_CONFIG" --learning-split "$LEARNING_SPLIT" \
    --partitions train val
}
cmd_candidates() {
  printf '%s\n' "$PY" "$PROJECT_ROOT/scripts/data/materialize_petct_component_candidates.py" \
    --learning-manifest "$AUDIT/tensors-rich.jsonl" \
    --output "$VISIBLE/candidates" --summary "$VISIBLE/candidates.jsonl"
}
cmd_three_manifests() {
  printf '%s\n' "$PY" "$PROJECT_ROOT/scripts/data/materialize_petct_r13_trajectory_5r_programs.py" \
    --source "$AUDIT/tensors-rich.jsonl" --learning-split "$LEARNING_SPLIT" \
    --lineage-receipt "$LINEAGE" --candidate-summary "$VISIBLE/candidates.jsonl" \
    --inference "$VISIBLE/episodes.jsonl" --labels "$LABEL/labels.jsonl" \
    --audit "$AUDIT/episodes.jsonl" --receipt "$MANIFEST_RECEIPT"
}
cmd_pointer_targets() {
  printf '%s\n' "$PY" "$PROJECT_ROOT/scripts/data/materialize_petct_component_targets.py" \
    --learning-manifest "$AUDIT/tensors-rich.jsonl" \
    --candidate-summary "$VISIBLE/candidates.jsonl" \
    --output "$LABEL/pointer-targets" --summary "$LABEL/pointer-targets.jsonl"
}
cmd_seal() {
  printf '%s\n' "$PY" "$TRAJECTORY_TOOL" seal \
    --lineage-receipt "$LINEAGE" --manifest-receipt "$MANIFEST_RECEIPT" \
    --inference-manifest "$VISIBLE/episodes.jsonl" \
    --label-manifest "$LABEL/labels.jsonl" --audit-manifest "$AUDIT/episodes.jsonl" \
    --rich-tensor-manifest "$AUDIT/tensors-rich.jsonl" \
    --candidate-summary "$VISIBLE/candidates.jsonl" \
    --pointer-summary "$LABEL/pointer-targets.jsonl" \
    --trajectories-summary "$AUDIT/trajectories.jsonl" \
    --output "$DATA_READY"
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
  emit_plan preflight cmd_preflight
  emit_plan lineage cmd_lineage
  emit_plan trajectories cmd_trajectories
  emit_plan tensors cmd_tensors
  emit_plan candidates cmd_candidates
  emit_plan three-manifests cmd_three_manifests
  emit_plan pointer-targets cmd_pointer_targets
  emit_plan seal cmd_seal
  exit 0
fi

# Frozen-input prechecks: every input must be a regular non-symlink file and
# the R13-main residuals must sit at the pinned audit-only locations.
for path in "$R13_MAIN_DATA_READY" "$RESIDUALS" "$RESIDUAL_READY" "$OOF_READY" "$LEARNING_SPLIT" "$EXPERIMENT_CONFIG" "$OFFICIAL_SIMULATOR" "$OFFICIAL_RUNTIME_MANIFEST"; do
  [[ -f "$path" && ! -L "$path" ]] || { echo "missing frozen R13 trajectory input: $path" >&2; exit 2; }
done
[[ ! -e "$RUN_ROOT" && ! -L "$RUN_ROOT" ]] || { echo "R13 trajectory run root already exists" >&2; exit 2; }

export TMPDIR=/mnt/HDD4/workspace/honor_degree/.tmp
export PIP_CACHE_DIR=/mnt/HDD4/workspace/honor_degree/.pip_cache
export CONDA_PKGS_DIRS=/mnt/HDD4/workspace/honor_degree/.conda_pkgs
export XDG_CACHE_HOME=/mnt/HDD4/workspace/honor_degree/.xdg_cache
export HF_HOME=/mnt/HDD4/workspace/honor_degree/.hf_home
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR" "$CONDA_PKGS_DIRS" "$XDG_CACHE_HOME" "$HF_HOME"
mkdir -p "$VISIBLE" "$LABEL" "$AUDIT" "$RUN_ROOT/logs" "$RUN_ROOT/state"

DONE=""; STAGE="preflight"
trap 'if [[ -z "$DONE" ]]; then printf "{\"status\":\"FAIL\",\"stage\":\"%s\",\"dataset_id\":\"R13-trajectory-5r\",\"source_m0_lineage\":\"M0_V6_FIVEFOLD_OOF\"}\n" "$STAGE" > "$STATE.fail"; rm -f "$STATE.running"; fi' EXIT
printf '{"status":"RUNNING","stage":"%s","dataset_id":"R13-trajectory-5r","source_m0_lineage":"M0_V6_FIVEFOLD_OOF","partitions":["train","val"]}\n' "$STAGE" > "$STATE.running"

run_step() {
  local name="$1"; shift
  STAGE="$name"
  printf '{"status":"RUNNING","stage":"%s"}\n' "$name" > "$RUN_ROOT/state/$name.running"
  if "$@" > "$RUN_ROOT/logs/$name.log" 2>&1; then
    mv "$RUN_ROOT/state/$name.running" "$RUN_ROOT/state/$name.done"
  else
    mv "$RUN_ROOT/state/$name.running" "$RUN_ROOT/state/$name.fail"
    return 1
  fi
}

CMD=()
while IFS= read -r line; do [[ -z "$line" ]] || CMD+=("$line"); done < <(cmd_preflight)
run_step preflight "${CMD[@]}"
CMD=()
while IFS= read -r line; do [[ -z "$line" ]] || CMD+=("$line"); done < <(cmd_lineage)
run_step lineage "${CMD[@]}"
CMD=()
while IFS= read -r line; do [[ -z "$line" ]] || CMD+=("$line"); done < <(cmd_trajectories)
run_step trajectories "${CMD[@]}"
CMD=()
while IFS= read -r line; do [[ -z "$line" ]] || CMD+=("$line"); done < <(cmd_tensors)
run_step tensors "${CMD[@]}"
CMD=()
while IFS= read -r line; do [[ -z "$line" ]] || CMD+=("$line"); done < <(cmd_candidates)
run_step candidates "${CMD[@]}"
CMD=()
while IFS= read -r line; do [[ -z "$line" ]] || CMD+=("$line"); done < <(cmd_three_manifests)
run_step three-manifests "${CMD[@]}"
CMD=()
while IFS= read -r line; do [[ -z "$line" ]] || CMD+=("$line"); done < <(cmd_pointer_targets)
run_step pointer-targets "${CMD[@]}"
CMD=()
while IFS= read -r line; do [[ -z "$line" ]] || CMD+=("$line"); done < <(cmd_seal)
run_step seal "${CMD[@]}"

printf '{"status":"PASS","dataset_id":"R13-trajectory-5r","source_m0_lineage":"M0_V6_FIVEFOLD_OOF","data_ready":"%s"}\n' "$DATA_READY" > "$STATE.done"
rm -f "$STATE.running"
DONE=1
