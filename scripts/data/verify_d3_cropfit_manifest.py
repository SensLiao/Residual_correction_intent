#!/usr/bin/env python3
"""Verify the crop-fit filtered manifest against the original D3 manifest.

Checks:
- kept + excluded == original row count, disjoint episode_ids, full coverage;
- kept rows are verbatim copies of the original rows;
- per-partition patient sets are preserved (union of kept + excluded);
- every exclusion record carries the frozen crop error and a positive overshoot;
- no partition=test rows anywhere.

Fails (non-zero) on any violation; prints a summary JSON otherwise.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Sequence

SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from common.petct_learning import load_jsonl  # noqa: E402

CROP_ERROR_MARKER = "authorized COMPLETE/LOCAL target exceeds frozen crop"


def _ids(rows: Sequence[Dict], key="episode_id"):
    return {str(row[key]) for row in rows}


def _load_exclusions(path: Path) -> list:
    # Zero exclusions is a legitimate outcome; load_jsonl rejects empty files.
    if path.exists() and path.stat().st_size > 0:
        return load_jsonl(path)
    return []


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-manifest", type=Path, required=True)
    parser.add_argument("--kept-manifest", type=Path, required=True)
    parser.add_argument("--exclusions", type=Path, required=True)
    args = parser.parse_args(argv)

    original = load_jsonl(args.original_manifest)
    kept = load_jsonl(args.kept_manifest)
    excluded = _load_exclusions(args.exclusions)

    assert any(row.get("partition") == "test" for row in original) is False, (
        "original manifest contains partition=test rows"
    )
    assert any(row.get("partition") == "test" for row in kept) is False
    assert any(
        rec["row"].get("partition") == "test" for rec in excluded
    ) is False

    original_by_id = {str(row["episode_id"]): row for row in original}
    assert len(original_by_id) == len(original), "duplicate episode_id in original"

    kept_ids = _ids(kept)
    excluded_ids = _ids([rec["row"] for rec in excluded])
    assert len(kept) + len(excluded) == len(original), (
        "count mismatch: %d kept + %d excluded != %d original"
        % (len(kept), len(excluded), len(original))
    )
    assert kept_ids.isdisjoint(excluded_ids), "kept/excluded episode_id overlap"
    assert kept_ids | excluded_ids == set(original_by_id), "coverage mismatch"

    for row in kept:
        assert row == original_by_id[str(row["episode_id"])], (
            "kept row %s is not a verbatim copy" % row["episode_id"]
        )

    partition_summary = {}
    for partition in sorted({row["partition"] for row in original}):
        original_patients = {
            str(row["patient_id"])
            for row in original
            if row["partition"] == partition
        }
        kept_patients = {
            str(row["patient_id"]) for row in kept if row["partition"] == partition
        }
        excluded_patients = {
            str(rec["row"]["patient_id"])
            for rec in excluded
            if rec["row"]["partition"] == partition
        }
        assert kept_patients | excluded_patients == original_patients, (
            "patient set changed in partition %s" % partition
        )
        partition_summary[partition] = {
            "original": len(
                [r for r in original if r["partition"] == partition]
            ),
            "kept": len([r for r in kept if r["partition"] == partition]),
            "excluded": len(
                [r for r in excluded if r["row"]["partition"] == partition]
            ),
            "patients": len(original_patients),
        }

    for rec in excluded:
        assert rec["error"] == CROP_ERROR_MARKER, "unexpected exclusion error"
        overshoot = rec["report"].get("overshoot_mm")
        assert overshoot is not None and overshoot > 0.0, (
            "excluded row %s has no positive overshoot"
            % rec["row"]["episode_id"]
        )

    excluded_goals: Dict[str, int] = {}
    for rec in excluded:
        goal = str(rec["row"]["goal"])
        excluded_goals[goal] = excluded_goals.get(goal, 0) + 1

    print(
        json.dumps(
            {
                "original_rows": len(original),
                "kept_rows": len(kept),
                "excluded_rows": len(excluded),
                "excluded_by_goal": excluded_goals,
                "partition_summary": partition_summary,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
