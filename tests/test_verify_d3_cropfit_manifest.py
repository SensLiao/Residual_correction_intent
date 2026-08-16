from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


PROJECT = Path(__file__).resolve().parents[1]
SCRIPTS = PROJECT / "scripts"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(SCRIPTS / "common"))
sys.path.insert(0, str(SCRIPTS / "data"))

import verify_d3_cropfit_manifest as verify  # noqa: E402


CROP_ERROR_MARKER = "authorized COMPLETE/LOCAL target exceeds frozen crop"


def _row(episode_id: str, patient_id: str = "p1", partition: str = "train") -> dict:
    return {
        "episode_id": episode_id,
        "patient_id": patient_id,
        "partition": partition,
        "goal": "ADD_SAME_LOCAL",
    }


def _write(tmp_path: Path, name: str, rows) -> Path:
    path = tmp_path / name
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    return path


def _exclusion(row: dict) -> dict:
    return {
        "row": row,
        "error": CROP_ERROR_MARKER,
        "report": {
            "crop_limit_mm": 95.625,
            "max_authorized_offset_mm": 120.0,
            "overshoot_mm": 24.375,
        },
    }


def _run(tmp_path, original, kept, excluded):
    return verify.main(
        [
            "--original-manifest",
            str(_write(tmp_path, "original.jsonl", original)),
            "--kept-manifest",
            str(_write(tmp_path, "kept.jsonl", kept)),
            "--exclusions",
            str(_write(tmp_path, "exclusions.jsonl", excluded)),
        ]
    )


def test_happy_path_preserves_counts_and_patients(tmp_path):
    rows = [_row("e1", "p1", "train"), _row("e2", "p2", "train"), _row("e3", "p3", "val")]
    excluded = [_exclusion(rows[1])]
    kept = [rows[0], rows[2]]
    assert _run(tmp_path, rows, kept, excluded) == 0


def test_count_mismatch_fails(tmp_path):
    rows = [_row("e1"), _row("e2")]
    with pytest.raises(AssertionError):
        _run(tmp_path, rows, [rows[0]], [])


def test_non_verbatim_kept_row_fails(tmp_path):
    rows = [_row("e1")]
    mutated = [dict(rows[0], goal="REMOVE_NEW_COMPLETE")]
    with pytest.raises(AssertionError):
        _run(tmp_path, rows, mutated, [])


def test_partition_test_rows_rejected(tmp_path):
    rows = [_row("e1", partition="test")]
    with pytest.raises(AssertionError):
        _run(tmp_path, rows, [rows[0]], [])


def test_exclusion_without_overshoot_fails(tmp_path):
    rows = [_row("e1"), _row("e2")]
    bad = {
        "row": rows[1],
        "error": CROP_ERROR_MARKER,
        "report": {"crop_limit_mm": 95.625, "overshoot_mm": None},
    }
    with pytest.raises(AssertionError):
        _run(tmp_path, rows, [rows[0]], [bad])


def test_patient_set_change_fails(tmp_path):
    rows = [_row("e1", "p1", "train"), _row("e2", "p2", "train")]
    excluded = [_exclusion(dict(rows[1], patient_id="p_other"))]
    with pytest.raises(AssertionError):
        _run(tmp_path, rows, [rows[0]], excluded)
