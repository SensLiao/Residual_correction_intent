#!/usr/bin/env python3
"""
Acceptance check for D-2026-08-05-02, run on a REAL materialised corpus.

What it is for
--------------
The project's own test suite already proves the v2 six-class chain works on synthetic
fixtures (587 passed / 9 skipped, 2026-08-05).  What no test covers is the actual
materialised corpus on the A6000: that every episode really carries its polarity, and
that an ADD episode and a REMOVE episode are not byte-identical inputs.  That is the
single claim `D-2026-08-05-02` has to retire, so it gets its own check on real bytes.

Canonical channel layout (config PETCT-ROUTE-A-EXPERIMENT-v2.0, NOT a signed single channel)
---------------------------------------------------------------------------------------------
  visible npz : visual[17,H,W] = PET z-2..z+2 | CT z-2..z+2 | M0 z-2..z+2 | cue_fg | cue_bg
                plus m0, scribble (UNSIGNED union of both polarities), cue_fg, cue_bg, spacing_xy
  P2T input   : all 17 channels
  editor input: 12 channels, signed cue derived as visual[15] - visual[16]

Checks
------
  A1 goal is one of the six legal joint goals and agrees with `operation`
  A2 npz carries every required key and visual has exactly 17 channels
  A3 cue_fg / cue_bg disjoint, union == scribble support, only the operation's own side filled
  A4 visual[15] == cue_fg and visual[16] == cue_bg
  A5 derived signed cue carries +1 on ADD support and -1 on REMOVE support
  A6 SEPARABILITY: the signed-value sets of ADD and REMOVE are disjoint across the corpus
  A7 mirror invariant: ADD authorized subset of GT\\M0; REMOVE authorized subset of M0\\GT
  A8 the cue lies inside the authorised target
  A9 HEADLINE: swapping only the polarity changes the tensor, per episode.  This is the
     direct refutation of "ADD and REMOVE produce byte-identical training input".

Usage
-----
  python corpus_polarity_acceptance.py --manifest <materialised manifest.jsonl> [--limit N] [--report out.json]
  python corpus_polarity_acceptance.py --self-test     # offline proof the checker works
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

LEGAL_GOALS = {
    "ADD_SAME_LOCAL", "REMOVE_SAME_LOCAL",
    "ADD_SAME_COMPLETE", "REMOVE_SAME_COMPLETE",
    "ADD_NEW_COMPLETE", "REMOVE_NEW_COMPLETE",
}
REQUIRED_KEYS = {"visual", "m0", "scribble", "cue_fg", "cue_bg", "spacing_xy"}
N_CHANNELS = 17
CUE_FG_CH, CUE_BG_CH = 15, 16


def check_corpus(rows: list[dict]) -> dict:
    fail: dict[str, list[str]] = collections.defaultdict(list)
    goal_counts: collections.Counter = collections.Counter()
    signs: dict[str, set] = collections.defaultdict(set)
    swap_changes = 0

    for row in rows:
        eid = str(row.get("episode_id"))
        goal = str(row.get("goal") or "")
        operation = str(row.get("operation") or "")
        goal_counts[goal] += 1

        # ---- A1 -----------------------------------------------------------
        if goal not in LEGAL_GOALS:
            fail["A1"].append(f"{eid}: illegal goal {goal!r}")
            continue
        if operation not in ("ADD", "REMOVE") or not goal.startswith(operation + "_"):
            fail["A1"].append(f"{eid}: operation {operation!r} disagrees with goal {goal!r}")
            continue

        with np.load(str(row["visible_npz"]), allow_pickle=False) as b:
            missing = REQUIRED_KEYS - set(b.files)
            if missing:
                fail["A2"].append(f"{eid}: visible npz missing {sorted(missing)}")
                continue
            visual = np.asarray(b["visual"], dtype=np.float32)
            scribble = np.asarray(b["scribble"]).astype(bool)
            fg = np.asarray(b["cue_fg"]).astype(bool)
            bg = np.asarray(b["cue_bg"]).astype(bool)
            m0 = np.asarray(b["m0"]).astype(bool)

        # ---- A2 -----------------------------------------------------------
        if visual.shape[0] != N_CHANNELS:
            fail["A2"].append(f"{eid}: visual has {visual.shape[0]} channels, want {N_CHANNELS}")
            continue

        # ---- A3 -----------------------------------------------------------
        if np.any(fg & bg):
            fail["A3"].append(f"{eid}: cue_fg and cue_bg overlap")
        if not np.array_equal(fg | bg, scribble):
            fail["A3"].append(f"{eid}: cue_fg|cue_bg != scribble support")
        if operation == "ADD" and (not fg.any() or bg.any()):
            fail["A3"].append(f"{eid}: ADD must fill cue_fg only")
        if operation == "REMOVE" and (not bg.any() or fg.any()):
            fail["A3"].append(f"{eid}: REMOVE must fill cue_bg only")

        # ---- A4 -----------------------------------------------------------
        if not np.array_equal(visual[CUE_FG_CH].astype(bool), fg):
            fail["A4"].append(f"{eid}: visual[{CUE_FG_CH}] != cue_fg")
        if not np.array_equal(visual[CUE_BG_CH].astype(bool), bg):
            fail["A4"].append(f"{eid}: visual[{CUE_BG_CH}] != cue_bg")

        # ---- A5 / A6 -------------------------------------------------------
        signed = visual[CUE_FG_CH] - visual[CUE_BG_CH]
        nz = signed[signed != 0]
        want = 1.0 if operation == "ADD" else -1.0
        if nz.size == 0:
            fail["A5"].append(f"{eid}: derived signed cue is empty")
        elif not np.all(nz == want):
            fail["A5"].append(f"{eid}: {operation} signed cue carries {sorted(set(nz.tolist()))}")
        signs[operation] |= set(np.unique(nz).tolist())

        # ---- A9: polarity swap must change the tensor -----------------------
        swapped = visual.copy()
        swapped[[CUE_FG_CH, CUE_BG_CH]] = swapped[[CUE_BG_CH, CUE_FG_CH]]
        if np.array_equal(swapped, visual):
            fail["A9"].append(f"{eid}: swapping polarity leaves the tensor unchanged")
        else:
            swap_changes += 1

        # ---- A7 / A8 -------------------------------------------------------
        ev = row.get("evaluation_npz")
        if ev:
            with np.load(str(ev), allow_pickle=False) as b:
                authorized = np.asarray(b["authorized"]).astype(bool)
                gt = np.asarray(b["gt"]).astype(bool)
            if operation == "ADD":
                if np.any(authorized & ~gt) or np.any(authorized & m0):
                    fail["A7"].append(f"{eid}: ADD authorized not inside GT\\M0")
            else:
                if np.any(authorized & ~m0) or np.any(authorized & gt):
                    fail["A7"].append(f"{eid}: REMOVE authorized not inside M0\\GT")
            if np.any(scribble & ~authorized):
                fail["A8"].append(f"{eid}: cue leaves the authorised target")

    # ---- A6 corpus-level separability --------------------------------------
    add_s, rem_s = signs.get("ADD", set()), signs.get("REMOVE", set())
    if add_s & rem_s:
        fail["A6"].append(f"ADD and REMOVE share signed values {sorted(add_s & rem_s)} "
                          "-> the two operations are STILL indistinguishable")
    if not add_s or not rem_s:
        fail["A6"].append(f"one operation absent (ADD={sorted(add_s)}, REMOVE={sorted(rem_s)}) "
                          "-> separability untested")

    return {
        "episodes_checked": len(rows),
        "goal_counts": dict(sorted(goal_counts.items(), key=lambda t: -t[1])),
        "signed_values_by_operation": {k: sorted(v) for k, v in signs.items()},
        "episodes_where_polarity_swap_changes_input": swap_changes,
        "failures": {k: {"count": len(v), "examples": v[:5]} for k, v in fail.items()},
        "all_checks_pass": not fail,
    }


# ----------------------------------------------------------------- self-test

def _fixture(tmp: Path, operation: str, *, break_polarity: bool = False) -> dict:
    """One tiny but structurally faithful episode."""
    rng = np.random.default_rng(0)
    h = w = 8
    m0 = np.zeros((h, w), np.uint8)
    m0[2:5, 2:5] = 1
    gt = np.zeros((h, w), np.uint8)
    gt[3:7, 3:7] = 1
    # ADD fixes GT\M0 ; REMOVE fixes M0\GT
    residual = (gt > 0) & ~(m0 > 0) if operation == "ADD" else (m0 > 0) & ~(gt > 0)
    authorized = residual.copy()
    cue = np.zeros((h, w), bool)
    idx = np.argwhere(authorized)
    cue[tuple(idx[0])] = True
    fg = cue if operation == "ADD" else np.zeros_like(cue)
    bg = cue if operation == "REMOVE" else np.zeros_like(cue)
    if break_polarity:                      # the v1 failure mode: polarity thrown away
        fg = cue.copy()
        bg = cue.copy()
    visual = np.concatenate([
        rng.normal(size=(15, h, w)).astype(np.float32),
        fg.astype(np.float32)[None],
        bg.astype(np.float32)[None],
    ], axis=0)
    eid = f"{operation}_ep"
    vis_p, ev_p = tmp / f"{eid}_vis.npz", tmp / f"{eid}_ev.npz"
    np.savez_compressed(vis_p, visual=visual, m0=m0, scribble=(fg | bg).astype(np.uint8),
                        cue_fg=fg.astype(np.uint8), cue_bg=bg.astype(np.uint8),
                        spacing_xy=np.asarray([2.0, 2.0], np.float32))
    np.savez_compressed(ev_p, target=authorized.astype(np.uint8),
                        authorized=authorized.astype(np.uint8), gt=gt)
    return {"episode_id": eid, "goal": f"{operation}_SAME_COMPLETE", "operation": operation,
            "visible_npz": str(vis_p), "evaluation_npz": str(ev_p)}


def self_test() -> int:
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        (tmp / "g").mkdir(parents=True, exist_ok=True)
        good = [_fixture(tmp / "g", op) for op in ("ADD", "REMOVE")]
        ok = check_corpus(good)
        print("GOOD corpus ->", json.dumps(
            {k: ok[k] for k in ("all_checks_pass", "signed_values_by_operation",
                                "episodes_where_polarity_swap_changes_input")},
            ensure_ascii=False))
        (tmp / "b").mkdir(parents=True, exist_ok=True)
        bad = [_fixture(tmp / "b", op, break_polarity=True) for op in ("ADD", "REMOVE")]
        ng = check_corpus(bad)
        print("BROKEN corpus (polarity discarded, the v1 failure mode) ->", json.dumps(
            {"all_checks_pass": ng["all_checks_pass"],
             "failed_checks": sorted(ng["failures"])}, ensure_ascii=False))

    if not ok["all_checks_pass"]:
        print("SELF-TEST FAILED: the checker rejects a correct corpus", file=sys.stderr)
        return 5
    if ng["all_checks_pass"]:
        print("SELF-TEST FAILED: the checker accepts a polarity-blind corpus", file=sys.stderr)
        return 5
    print("\nSELF-TEST PASS — accepts correct v2 bytes, rejects the v1 failure mode.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--report", type=Path)
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()

    if a.self_test:
        return self_test()
    if not a.manifest:
        ap.error("--manifest is required unless --self-test")

    rows = [json.loads(x) for x in a.manifest.read_text(encoding="utf-8").splitlines() if x.strip()]
    if a.limit:
        rows = rows[: a.limit]
    rep = check_corpus(rows)
    print(json.dumps(rep, indent=2, ensure_ascii=False))
    if a.report:
        a.report.parent.mkdir(parents=True, exist_ok=True)
        a.report.write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")
    if not rep["all_checks_pass"]:
        print("\nFAILED", file=sys.stderr)
        return 4
    print("\nPASS — every episode carries its polarity and ADD/REMOVE are not byte-identical.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
