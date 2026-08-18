#!/usr/bin/env bash
# Build R13-main from one explicit M0_V6_FIVEFOLD_OOF receipt. TRAIN/VAL only.
set -euo pipefail

usage() {
  echo "usage: $0 --run-root DIR --oof-ready FILE --case-manifest FILE --learning-split FILE --experiment-config FILE --official-simulator FILE --official-runtime-manifest FILE" >&2
}

RUN_ROOT=""; OOF_READY=""; CASE_MANIFEST=""; LEARNING_SPLIT=""
EXPERIMENT_CONFIG=""; OFFICIAL_SIMULATOR=""; OFFICIAL_RUNTIME_MANIFEST=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-root) RUN_ROOT="${2:-}"; shift 2;;
    --oof-ready) OOF_READY="${2:-}"; shift 2;;
    --case-manifest) CASE_MANIFEST="${2:-}"; shift 2;;
    --learning-split) LEARNING_SPLIT="${2:-}"; shift 2;;
    --experiment-config) EXPERIMENT_CONFIG="${2:-}"; shift 2;;
    --official-simulator) OFFICIAL_SIMULATOR="${2:-}"; shift 2;;
    --official-runtime-manifest) OFFICIAL_RUNTIME_MANIFEST="${2:-}"; shift 2;;
    *) usage; exit 2;;
  esac
done
for value in "$RUN_ROOT" "$OOF_READY" "$CASE_MANIFEST" "$LEARNING_SPLIT" "$EXPERIMENT_CONFIG" "$OFFICIAL_SIMULATOR" "$OFFICIAL_RUNTIME_MANIFEST"; do
  [[ -n "$value" ]] || { usage; exit 2; }
done

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
RUN_BASE="$PROJECT_ROOT/route_a/runs"
RUN_ROOT=$(readlink -m "$RUN_ROOT")
case "$RUN_ROOT" in "$RUN_BASE"/PETCT-R13-MAIN-*) ;; *) echo "invalid R13 run root" >&2; exit 2;; esac
[[ ! -e "$RUN_ROOT" && ! -L "$RUN_ROOT" ]] || { echo "R13 run root already exists" >&2; exit 2; }
for path in "$OOF_READY" "$CASE_MANIFEST" "$LEARNING_SPLIT" "$EXPERIMENT_CONFIG" "$OFFICIAL_SIMULATOR" "$OFFICIAL_RUNTIME_MANIFEST"; do
  [[ -f "$path" && ! -L "$path" ]] || { echo "missing explicit R13 input: $path" >&2; exit 2; }
done

PY="$PROJECT_ROOT/envs/petct_nnunet_v281/bin/python"
LINEAGE_TOOL="$PROJECT_ROOT/scripts/common/petct_mainline_lineage.py"
DATA="$RUN_ROOT/R13-main"
VISIBLE="$DATA/inference-visible"
LABEL="$DATA/label-only"
AUDIT="$DATA/audit-only"
LINEAGE="$DATA/lineage-receipt.json"
DATA_READY="$DATA/data-ready.json"
STATE="$RUN_ROOT/state/r13_main"

export TMPDIR=/mnt/HDD4/zlei0805/honor_degree/.tmp
export PIP_CACHE_DIR=/mnt/HDD4/zlei0805/honor_degree/.pip_cache
export CONDA_PKGS_DIRS=/mnt/HDD4/zlei0805/honor_degree/.conda_pkgs
export XDG_CACHE_HOME=/mnt/HDD4/zlei0805/honor_degree/.xdg_cache
export HF_HOME=/mnt/HDD4/zlei0805/honor_degree/.hf_home
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR" "$CONDA_PKGS_DIRS" "$XDG_CACHE_HOME" "$HF_HOME"
mkdir -p "$VISIBLE" "$LABEL" "$AUDIT" "$RUN_ROOT/logs" "$RUN_ROOT/state"

DONE=""; STAGE="issue_lineage"
trap 'if [[ -z "$DONE" ]]; then printf "{\"status\":\"FAIL\",\"stage\":\"%s\",\"source_m0_lineage\":\"M0_V6_FIVEFOLD_OOF\"}\n" "$STAGE" > "$STATE.fail"; rm -f "$STATE.running"; fi' EXIT
printf '{"status":"RUNNING","stage":"%s","source_m0_lineage":"M0_V6_FIVEFOLD_OOF","partitions":["train","val"]}\n' "$STAGE" > "$STATE.running"

"$PY" "$LINEAGE_TOOL" issue --oof-ready "$OOF_READY" \
  --learning-split "$LEARNING_SPLIT" --experiment-config "$EXPERIMENT_CONFIG" \
  --output "$LINEAGE" > "$RUN_ROOT/logs/issue_lineage.log" 2>&1

STAGE=residuals
"$PY" "$PROJECT_ROOT/scripts/data/build_petct_residual_manifest.py" \
  --oof-ready "$OOF_READY" --learning-split "$LEARNING_SPLIT" \
  --experiment-config "$EXPERIMENT_CONFIG" --case-manifest "$CASE_MANIFEST" \
  --partitions train val --output-dir "$AUDIT/residual-masks" \
  --output-manifest "$AUDIT/residuals.jsonl" \
  --ready-receipt "$AUDIT/RESIDUAL_READY.json" \
  > "$RUN_ROOT/logs/residuals.log" 2>&1

STAGE=single_round_episodes
"$PY" "$PROJECT_ROOT/scripts/data/build_petct_scribble_dataset.py" \
  --residual-manifest "$AUDIT/residuals.jsonl" \
  --residual-ready "$AUDIT/RESIDUAL_READY.json" \
  --official-simulator "$OFFICIAL_SIMULATOR" \
  --official-runtime-manifest "$OFFICIAL_RUNTIME_MANIFEST" \
  --experiment-config "$EXPERIMENT_CONFIG" --lane natural \
  --oof-ready "$OOF_READY" --learning-split "$LEARNING_SPLIT" \
  --partitions train val --strategy-mode primary --seed 42 \
  --visible-root "$VISIBLE/episode-documents" \
  --evaluation-root "$AUDIT/episode-documents" \
  --authorized-root "$AUDIT/authorized-masks" \
  --output-manifest "$AUDIT/episodes-rich.jsonl" \
  --exclusions "$AUDIT/episode-exclusions.jsonl" \
  --ready-receipt "$AUDIT/EPISODES_READY.json" \
  > "$RUN_ROOT/logs/episodes.log" 2>&1

STAGE=tensors
"$PY" "$PROJECT_ROOT/scripts/data/materialize_petct_learning_tensors.py" \
  --episode-manifest "$AUDIT/episodes-rich.jsonl" \
  --visible-root "$VISIBLE/tensors" --evaluation-root "$AUDIT/tensors" \
  --output-manifest "$AUDIT/tensors-rich.jsonl" \
  --experiment-config "$EXPERIMENT_CONFIG" --learning-split "$LEARNING_SPLIT" \
  --partitions train val > "$RUN_ROOT/logs/tensors.log" 2>&1

STAGE=candidates
"$PY" "$PROJECT_ROOT/scripts/data/materialize_petct_component_candidates.py" \
  --learning-manifest "$AUDIT/tensors-rich.jsonl" \
  --output "$VISIBLE/candidates" --summary "$VISIBLE/candidates.jsonl" \
  > "$RUN_ROOT/logs/candidates.log" 2>&1

STAGE=three_manifests
"$PY" "$PROJECT_ROOT/scripts/data/materialize_petct_program_manifests.py" \
  --source "$AUDIT/tensors-rich.jsonl" --learning-split "$LEARNING_SPLIT" \
  --lineage-receipt "$LINEAGE" --candidate-summary "$VISIBLE/candidates.jsonl" \
  --inference "$VISIBLE/episodes.jsonl" --labels "$LABEL/labels.jsonl" \
  --audit "$AUDIT/episodes.jsonl" --receipt "$DATA/program-manifest-receipt.json" \
  > "$RUN_ROOT/logs/three_manifests.log" 2>&1

STAGE=pointer_targets
"$PY" "$PROJECT_ROOT/scripts/data/materialize_petct_component_targets.py" \
  --learning-manifest "$AUDIT/tensors-rich.jsonl" \
  --candidate-summary "$VISIBLE/candidates.jsonl" \
  --output "$LABEL/pointer-targets" --summary "$LABEL/pointer-targets.jsonl" \
  > "$RUN_ROOT/logs/pointer_targets.log" 2>&1

STAGE=seal
"$PY" "$LINEAGE_TOOL" seal --lineage-receipt "$LINEAGE" \
  --manifest-receipt "$DATA/program-manifest-receipt.json" \
  --inference-manifest "$VISIBLE/episodes.jsonl" \
  --label-manifest "$LABEL/labels.jsonl" --audit-manifest "$AUDIT/episodes.jsonl" \
  --rich-tensor-manifest "$AUDIT/tensors-rich.jsonl" \
  --candidate-summary "$VISIBLE/candidates.jsonl" \
  --pointer-summary "$LABEL/pointer-targets.jsonl" --output "$DATA_READY" \
  > "$RUN_ROOT/logs/seal.log" 2>&1

printf '{"status":"PASS","source_m0_lineage":"M0_V6_FIVEFOLD_OOF","dataset_id":"R13-main-single-round","data_ready":"%s"}\n' "$DATA_READY" > "$STATE.done"
rm -f "$STATE.running"
DONE=1
