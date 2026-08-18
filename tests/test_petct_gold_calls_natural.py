"""Contract tests for the natural-lane gold oracle-calls builder (B1 lane fix).

The frozen ``render_petct_gold_program_calls_v3.py`` calls
``load_label_manifest`` with ``require_matched_groups=True`` and therefore
cannot consume R13-main natural labels (no ``matched_state_group_id``).  The
natural variant reuses the same contract helpers and derive loop with
``require_matched_groups=False``; these tests pin that behavior plus its
equivalence to the frozen builder on matched labels.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
import sys  # noqa: E402

if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from evaluation import (  # noqa: E402
    render_petct_gold_program_calls_natural as natural,
    render_petct_gold_program_calls_v3 as frozen,
)
from evaluation.render_petct_gold_program_calls_v3 import (  # noqa: E402
    ORACLE_SCHEMA,
)


def _label(episode_id, goal, operation, partition="val", extra=None):
    row = {
        "schema_version": "PETCT-PROGRAM-LABEL-MANIFEST-v1.0",
        "episode_id": episode_id,
        "case_id": "case-" + episode_id,
        "patient_id": "patient-" + episode_id,
        "partition": partition,
        "goal": goal,
        "operation": operation,
        "evaluation_npz": "npz-" + episode_id,
        "evaluation_sha256": "e" * 64,
        "learning_split_sha256": "l" * 64,
    }
    if extra:
        row.update(extra)
    return row


def _candidate(episode_id, components, cue_hit=None):
    return {
        "episode_id": episode_id,
        "components": [
            {"component_key": key, "distance_from_cue_mm": float(distance)}
            for key, distance in components
        ],
        "cue_hit_component_position": cue_hit,
    }


def _pointer(episode_id, positions):
    return {"episode_id": episode_id, "pointer_target_positions": positions}


def _write_tree(root: Path, sidecar_name: str, records) -> Path:
    directory = root / sidecar_name
    directory.mkdir()
    for record in records:
        (directory / (record["episode_id"] + ".json")).write_text(
            json.dumps(record), encoding="utf-8"
        )
    return directory


def _render(builder, tmp_path: Path, labels, candidates, pointers):
    tmp_path.mkdir(parents=True, exist_ok=True)
    labels_path = tmp_path / "labels.jsonl"
    with labels_path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in labels:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    candidates_dir = _write_tree(tmp_path, "candidates", candidates)
    pointers_dir = _write_tree(tmp_path, "pointer-targets", pointers)
    output = tmp_path / "oracle-calls.jsonl"
    receipt = tmp_path / "oracle-calls.receipt.json"
    rc = builder.main(
        [
            "--labels",
            str(labels_path),
            "--candidates",
            str(candidates_dir),
            "--pointer-targets",
            str(pointers_dir),
            "--partition",
            "val",
            "--output",
            str(output),
            "--receipt",
            str(receipt),
        ]
    )
    assert rc == 0
    calls = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    return calls, json.loads(receipt.read_text(encoding="utf-8"))


NATURAL_LABELS = [
    _label("e1", "ADD_SAME_LOCAL", "ADD"),
    _label("e2", "ADD_NEW_COMPLETE", "ADD"),
    _label("e3", "REMOVE_SAME_COMPLETE", "REMOVE"),
]
NATURAL_CANDIDATES = [
    _candidate(
        "e1",
        [("c0", 1.0), ("c1", 3.0), ("c2", 10.0)],
    ),
    _candidate("e2", [("c0", 1.0)]),
    _candidate(
        "e3",
        [("c0", 1.0), ("c1", 4.0)],
        cue_hit=1,
    ),
]
NATURAL_POINTERS = [
    _pointer("e1", [0, 1]),
    _pointer("e2", []),
    _pointer("e3", []),
]


def test_natural_builder_accepts_labels_without_matched_groups(tmp_path):
    calls, receipt = _render(
        natural, tmp_path, NATURAL_LABELS, NATURAL_CANDIDATES, NATURAL_POINTERS
    )
    assert receipt["status"] == "PASS"
    assert receipt["call_count"] == 3
    assert receipt.get("natural_lane") is True
    by_episode = {call["episode_id"]: call for call in calls}
    assert all(call["schema_version"] == ORACLE_SCHEMA for call in calls)
    # Derived call must round-trip the label goal (family names belong to the
    # frozen contract mapping, so assert the goal invariant, not names).
    assert {call["episode_id"]: call["goal"] for call in calls} == {
        "e1": "ADD_SAME_LOCAL",
        "e2": "ADD_NEW_COMPLETE",
        "e3": "REMOVE_SAME_COMPLETE",
    }
    # ADD existing -> nearest gold-positive component by cue distance
    assert by_episode["e1"]["operand"] == "c0"
    # ADD new -> NEW_CUE sentinel
    assert by_episode["e2"]["operand"] == "NEW_CUE"
    # REMOVE -> deterministic cue-hit component
    assert by_episode["e3"]["operand"] == "c1"


def test_natural_builder_matches_frozen_on_matched_labels(tmp_path):
    # The frozen builder validates matched-group triplets; build a legal
    # one-operation ADD triplet so both builders run on identical input.
    labels = [
        _label("m1", "ADD_SAME_LOCAL", "ADD", extra={"matched_state_group_id": "g1"}),
        _label("m2", "ADD_SAME_COMPLETE", "ADD", extra={"matched_state_group_id": "g1"}),
        _label("m3", "ADD_NEW_COMPLETE", "ADD", extra={"matched_state_group_id": "g1"}),
    ]
    candidates = [
        _candidate("m1", [("c0", 1.0), ("c1", 3.0)]),
        _candidate("m2", [("c0", 1.0), ("c1", 3.0)]),
        _candidate("m3", [("c0", 1.0)]),
    ]
    pointers = [_pointer("m1", [0, 1]), _pointer("m2", [0]), _pointer("m3", [])]
    frozen_calls, _ = _render(frozen, tmp_path / "frozen", labels, candidates, pointers)
    natural_calls, _ = _render(natural, tmp_path / "natural", labels, candidates, pointers)
    assert frozen_calls == natural_calls


def test_natural_builder_fails_closed_on_missing_candidate(tmp_path):
    with pytest.raises(Exception):
        _render(
            natural,
            tmp_path,
            NATURAL_LABELS,
            NATURAL_CANDIDATES[:-1],
            NATURAL_POINTERS,
        )


def test_natural_builder_refuses_existing_output(tmp_path):
    labels_path = tmp_path / "labels.jsonl"
    with labels_path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in NATURAL_LABELS:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    candidates_dir = _write_tree(tmp_path, "candidates", NATURAL_CANDIDATES)
    pointers_dir = _write_tree(tmp_path, "pointer-targets", NATURAL_POINTERS)
    output = tmp_path / "oracle-calls.jsonl"
    output.write_text("existing", encoding="utf-8")
    receipt = tmp_path / "receipt.json"
    with pytest.raises(SystemExit):
        natural.main(
            [
                "--labels",
                str(labels_path),
                "--candidates",
                str(candidates_dir),
                "--pointer-targets",
                str(pointers_dir),
                "--partition",
                "val",
                "--output",
                str(output),
                "--receipt",
                str(receipt),
            ]
        )
