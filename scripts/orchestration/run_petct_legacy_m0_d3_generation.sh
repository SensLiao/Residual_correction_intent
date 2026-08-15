#!/usr/bin/env bash
set -euo pipefail

P=/mnt/HDD4/workspace/honor_degree/projects/petct_textual_intent
PY="$P/envs/petct_nnunet_v281/bin/python"
RUN="$P/route_a/runs/PETCT-LEGACY-M0-D123-20260809-R3"

test ! -e "$RUN"
mkdir -p "$RUN/artifacts/d3" "$RUN/logs" "$RUN/status"
cd "$P"

fail_stage="generation"
on_failure() {
  exit_code=$?
  printf 'stage=%s\nexit_code=%s\n' "$fail_stage" "$exit_code" \
    >"$RUN/status/D3_FAILED"
  exit "$exit_code"
}
trap on_failure ERR

nice -n 10 ionice -c2 -n7 "$PY" scripts/data/build_petct_legacy_m0_d3_corpus.py \
  --d2-manifest "$P/route_a/runs/PETCT-TRAIN-20260807-R1/artifacts/episodes_5r_ok.jsonl" \
  --expected-d2-manifest-sha256 5ab06af11259998739a4a0bb92c0f6a7b34700a5479cf0eb2708aadbd32116e0 \
  --legacy-oof-ready-sha256 4009a8b0072eac55a52ba7441dbcb57175e3e368437e7a3a1f42830a471ba710 \
  --experiment-config "$P/configs/petct_route_a_experiment.json" \
  --learning-split "$P/records/data_readiness/psma_v3_learning_split_20260718.json" \
  --official-simulator "$P/upstream/autoPETV/interactive/simulate_scribbles.py" \
  --official-runtime-manifest "$P/protocols/autopetv_protocol_runtime.json" \
  --official-commit 4a2026866bfacc812492cfc7e6a8c54ac3c4f703 \
  --seed 42 --partitions train val \
  --visible-root "$RUN/artifacts/d3/visible_documents" \
  --evaluation-root "$RUN/artifacts/d3/evaluation_documents" \
  --output-manifest "$RUN/artifacts/d3/episode_manifest.jsonl" \
  --exclusions "$RUN/artifacts/d3/exclusions.jsonl" \
  --ready-receipt "$RUN/artifacts/d3/D3_READY.json" \
  >"$RUN/logs/d3-generate.log" 2>&1
printf 'DONE\n' >"$RUN/status/D3_GENERATION_DONE"

fail_stage="materialization"
nice -n 10 ionice -c2 -n7 "$PY" scripts/data/materialize_petct_learning_tensors.py \
  --episode-manifest "$RUN/artifacts/d3/episode_manifest.jsonl" \
  --visible-root "$RUN/artifacts/d3/learning-visible" \
  --evaluation-root "$RUN/artifacts/d3/learning-evaluation" \
  --output-manifest "$RUN/artifacts/d3/learning_tensors.jsonl" \
  --experiment-config "$P/configs/petct_route_a_experiment.json" \
  --learning-split "$P/records/data_readiness/psma_v3_learning_split_20260718.json" \
  --partitions train val \
  >"$RUN/logs/d3-materialize.log" 2>&1
printf 'DONE\n' >"$RUN/status/D3_MATERIALIZATION_DONE"

sha256sum \
  "$RUN/artifacts/d3/episode_manifest.jsonl" \
  "$RUN/artifacts/d3/exclusions.jsonl" \
  "$RUN/artifacts/d3/D3_READY.json" \
  "$RUN/artifacts/d3/learning_tensors.jsonl" \
  >"$RUN/status/D3_OUTPUT_SHA256SUMS.txt"
printf 'DONE\n' >"$RUN/status/D3_DONE"
trap - ERR
