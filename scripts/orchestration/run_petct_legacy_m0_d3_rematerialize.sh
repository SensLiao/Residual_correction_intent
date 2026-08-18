#!/usr/bin/env bash
# D-2026-08-13 Director option A (generation-side alignment):
# probe the frozen materializer over every D3 triplet, exclude rows whose
# authorized COMPLETE/LOCAL target exceeds the frozen crop, then materialize
# the kept manifest.  The frozen contract (config / crop / materializer) is
# never modified.  No locked-test access; train/val partitions only.
set -euo pipefail

echo "HISTORICAL_ONLY: D3 rematerialization is deprecated by the R13/M0-v6 cutover." >&2
exit 64

P=/mnt/HDD4/workspace/honor_degree/projects/petct_textual_intent
PY="$P/envs/petct_nnunet_v281/bin/python"
R3="$P/route_a/runs/PETCT-LEGACY-M0-D123-20260809-R3"
RUN="$P/route_a/runs/PETCT-LEGACY-M0-D123-20260813-R4"
CONFIG="$P/configs/petct_route_a_experiment.json"
SPLIT="$P/records/data_readiness/psma_v3_learning_split_20260718.json"
MANIFEST="$R3/artifacts/d3/episode_manifest.jsonl"
CROP_ERROR_MARKER="authorized COMPLETE/LOCAL target exceeds frozen crop"

test ! -e "$RUN"
mkdir -p "$RUN/artifacts/d3/smoke" "$RUN/logs" "$RUN/status" "$RUN/.tmp"
export TMPDIR="$RUN/.tmp"
export XDG_CACHE_HOME="$RUN/.tmp/cache"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME"
cd "$P"

fail_stage="probe_smoke"
on_failure() {
  exit_code=$?
  printf 'stage=%s\nexit_code=%s\n' "$fail_stage" "$exit_code" \
    >"$RUN/status/D3_MATERIALIZE_V2_FAILED"
  exit "$exit_code"
}
trap on_failure ERR

# ---------------------------------------------------------------- smoke ----
# 1 row x3 and 3 rows x3; outputs must be sha256-identical across repeats.
run_probe() {
  local n="$1" tag="$2"
  nice -n 10 ionice -c2 -n7 "$PY" scripts/data/probe_episode_crop_fit.py \
    --episode-manifest "$MANIFEST" \
    --experiment-config "$CONFIG" \
    --output-manifest "$RUN/artifacts/d3/smoke/${tag}_kept.jsonl" \
    --exclusions "$RUN/artifacts/d3/smoke/${tag}_exclusions.jsonl" \
    --max-rows "$n" \
    --temp-root "$RUN/artifacts/d3/smoke/tmp" \
    >>"$RUN/logs/d3-probe-smoke.log" 2>&1
}
for n in 1 3; do
  for r in 1 2 3; do
    run_probe "$n" "n${n}_r${r}"
  done
done
for n in 1 3; do
  for kind in kept exclusions; do
    sha1=$(sha256sum "$RUN/artifacts/d3/smoke/n${n}_r1_${kind}.jsonl" | awk '{print $1}')
    sha2=$(sha256sum "$RUN/artifacts/d3/smoke/n${n}_r2_${kind}.jsonl" | awk '{print $1}')
    sha3=$(sha256sum "$RUN/artifacts/d3/smoke/n${n}_r3_${kind}.jsonl" | awk '{print $1}')
    test "$sha1" = "$sha2" && test "$sha2" = "$sha3"
  done
done
printf 'DONE\n' >"$RUN/status/D3_PROBE_SMOKE_OK"

# ------------------------------------------------------------- full probe ---
fail_stage="probe_full"
nice -n 10 ionice -c2 -n7 "$PY" scripts/data/probe_episode_crop_fit.py \
  --episode-manifest "$MANIFEST" \
  --experiment-config "$CONFIG" \
  --output-manifest "$RUN/artifacts/d3/episode_manifest_cropfit.jsonl" \
  --exclusions "$RUN/artifacts/d3/crop_exclusions.jsonl" \
  --temp-root "$RUN/artifacts/d3/probe_tmp" \
  >"$RUN/logs/d3-probe-full.log" 2>&1
printf 'DONE\n' >"$RUN/status/D3_PROBE_DONE"

# ------------------------------------------------ probe/materializer xcheck ---
# One excluded row must make the real materializer raise the frozen crop error;
# one kept row must pass the frozen materializer end to end.
fail_stage="crosscheck"
"$PY" -c 'import json,sys
kept,excl,out_keep,out_excl=sys.argv[1:5]
with open(excl,encoding="utf-8") as f:
    ex=[json.loads(l) for l in f if l.strip()]
with open(kept,encoding="utf-8") as f:
    keep_row=json.loads(f.readline())
with open(out_keep,"w",encoding="utf-8",newline="\n") as f:
    f.write(json.dumps(keep_row,sort_keys=True)+"\n")
if ex:
    with open(out_excl,"w",encoding="utf-8",newline="\n") as f:
        f.write(json.dumps(ex[0]["row"],sort_keys=True)+"\n")
else:
    open(out_excl,"w").close()
print("xcheck rows prepared; exclusions=%d" % len(ex))' \
  "$RUN/artifacts/d3/episode_manifest_cropfit.jsonl" \
  "$RUN/artifacts/d3/crop_exclusions.jsonl" \
  "$RUN/artifacts/d3/xcheck_keep_row.jsonl" \
  "$RUN/artifacts/d3/xcheck_excluded_row.jsonl"

# excluded row: materializer MUST raise the frozen crop error (fail-closed)
if [[ -s "$RUN/artifacts/d3/xcheck_excluded_row.jsonl" ]]; then
  set +e
  "$PY" scripts/data/materialize_petct_learning_tensors.py \
    --episode-manifest "$RUN/artifacts/d3/xcheck_excluded_row.jsonl" \
    --visible-root "$RUN/artifacts/d3/xcheck_excl_visible" \
    --evaluation-root "$RUN/artifacts/d3/xcheck_excl_evaluation" \
    --output-manifest "$RUN/artifacts/d3/xcheck_excl_out.jsonl" \
    --experiment-config "$CONFIG" \
    --learning-split "$SPLIT" \
    --partitions train val \
    >"$RUN/logs/d3-xcheck-excluded.log" 2>&1
  rc=$?
  set -e
  test "$rc" -ne 0
  grep -q "$CROP_ERROR_MARKER" "$RUN/logs/d3-xcheck-excluded.log"
  printf 'EXCLUDED_ROW_RAISES_CROP_ERROR\n' >"$RUN/status/D3_XCHECK_EXCLUDED_OK"
else
  printf 'NO_EXCLUSIONS_SKIPPED\n' >"$RUN/status/D3_XCHECK_EXCLUDED_OK"
fi

# kept row: materializer MUST succeed end to end
"$PY" scripts/data/materialize_petct_learning_tensors.py \
  --episode-manifest "$RUN/artifacts/d3/xcheck_keep_row.jsonl" \
  --visible-root "$RUN/artifacts/d3/xcheck_keep_visible" \
  --evaluation-root "$RUN/artifacts/d3/xcheck_keep_evaluation" \
  --output-manifest "$RUN/artifacts/d3/xcheck_keep_out.jsonl" \
  --experiment-config "$CONFIG" \
  --learning-split "$SPLIT" \
  --partitions train val \
  >"$RUN/logs/d3-xcheck-kept.log" 2>&1
printf 'KEPT_ROW_MATERIALIZES\n' >"$RUN/status/D3_XCHECK_KEPT_OK"

# ---------------------------------------------------------- materialization ---
fail_stage="materialization"
nice -n 10 ionice -c2 -n7 "$PY" scripts/data/materialize_petct_learning_tensors.py \
  --episode-manifest "$RUN/artifacts/d3/episode_manifest_cropfit.jsonl" \
  --visible-root "$RUN/artifacts/d3/learning-visible" \
  --evaluation-root "$RUN/artifacts/d3/learning-evaluation" \
  --output-manifest "$RUN/artifacts/d3/learning_tensors.jsonl" \
  --experiment-config "$CONFIG" \
  --learning-split "$SPLIT" \
  --partitions train val \
  >"$RUN/logs/d3-materialize.log" 2>&1
printf 'DONE\n' >"$RUN/status/D3_MATERIALIZATION_DONE"

# ------------------------------------------------------------ verification ---
fail_stage="verification"
"$PY" scripts/data/verify_d3_cropfit_manifest.py \
  --original-manifest "$MANIFEST" \
  --kept-manifest "$RUN/artifacts/d3/episode_manifest_cropfit.jsonl" \
  --exclusions "$RUN/artifacts/d3/crop_exclusions.jsonl" \
  >"$RUN/logs/d3-verify.log" 2>&1

sha256sum \
  "$RUN/artifacts/d3/episode_manifest_cropfit.jsonl" \
  "$RUN/artifacts/d3/crop_exclusions.jsonl" \
  "$RUN/artifacts/d3/learning_tensors.jsonl" \
  >"$RUN/status/D3_OUTPUT_SHA256SUMS.txt"
printf 'DONE\n' >"$RUN/status/D3_MATERIALIZE_V2_DONE"
trap - ERR
