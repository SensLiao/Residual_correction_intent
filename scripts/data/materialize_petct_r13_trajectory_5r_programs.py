#!/usr/bin/env python3
"""Five-round variant of the v3 program-manifest materializer.

Reuses the frozen ``split_rows`` three-lane split and the visible firewall
unchanged.  The five-round identity binding stamps every row with
dataset_id R13-trajectory-5r, the explicit trajectory_id (label/audit lanes),
and round_index 0..4; the inference lane keeps the frozen R13-main allowed
field set so the generic visible-lane datasets accept the rows.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from common.petct_mainline_lineage import (  # noqa: E402
    MAINLINE_SOURCE,
    file_record,
)
from common.petct_program_learning import (  # noqa: E402
    LearningContractError,
    _sha256_file,
    load_jsonl,
    validate_training_split,
)
from common.petct_trajectory_lineage import (  # noqa: E402
    TRAJECTORY_DATASET_ID,
    TRAJECTORY_PROGRAM_RECEIPT_SCHEMA,
    TRAJECTORY_ROUND_LIMIT,
    TRAJECTORY_STATUSES,
    validate_r13_trajectory_rows,
    validate_trajectory_lineage_receipt,
)
from data.materialize_petct_program_manifests import (  # noqa: E402
    _assert_visible_value_safe,
    _canonical_sha,
    _write_jsonl_exclusive,
    split_rows,
)


def bind_trajectory_identity(
    source_rows: Sequence[Mapping[str, Any]],
    inference_rows: list[dict],
    label_rows: list[dict],
    audit_rows: list[dict],
) -> None:
    """Attach the explicit five-round identity to every program row."""

    for source, inference, label, audit in zip(
        source_rows, inference_rows, label_rows, audit_rows
    ):
        trajectory_id = str(source.get("trajectory_id") or "")
        if not trajectory_id:
            raise LearningContractError("trajectory source row lacks trajectory_id")
        round_index = source.get("round_index")
        if (
            isinstance(round_index, bool)
            or not isinstance(round_index, int)
            or round_index not in range(TRAJECTORY_ROUND_LIMIT)
        ):
            raise LearningContractError("trajectory source row has invalid round_index")
        round_count = source.get("round_count")
        if (
            isinstance(round_count, bool)
            or not isinstance(round_count, int)
            or not 1 <= round_count <= TRAJECTORY_ROUND_LIMIT
        ):
            raise LearningContractError("trajectory source row has invalid round_count")
        if int(round_index) >= int(round_count):
            raise LearningContractError("trajectory round_index exceeds round_count")
        status = str(source.get("trajectory_status") or "")
        if status not in TRAJECTORY_STATUSES:
            raise LearningContractError("trajectory source row has invalid status")
        strategy = str(source.get("strategy") or "")
        inference.update(
            {
                "dataset_id": TRAJECTORY_DATASET_ID,
                "source_m0_lineage": MAINLINE_SOURCE,
                "round_index": int(round_index),
                "scribble_count": 1,
            }
        )
        label.update(
            {
                "dataset_id": TRAJECTORY_DATASET_ID,
                "source_m0_lineage": MAINLINE_SOURCE,
                "episode_family_id": trajectory_id,
                "trajectory_id": trajectory_id,
                "round_index": int(round_index),
                "round_count": int(round_count),
                "trajectory_status": status,
                "scribble_count": 1,
                "strategy": strategy,
            }
        )
        if source.get("termination_reason") is not None:
            label["termination_reason"] = str(source["termination_reason"])
        audit.update(
            {
                "dataset_id": TRAJECTORY_DATASET_ID,
                "source_m0_lineage": MAINLINE_SOURCE,
                "episode_family_id": trajectory_id,
                "trajectory_id": trajectory_id,
                "round_index": int(round_index),
                "round_count": int(round_count),
                "trajectory_status": status,
                "scribble_count": 1,
                "strategy": strategy,
            }
        )
    validate_r13_trajectory_rows(label_rows)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--learning-split", type=Path, required=True)
    parser.add_argument("--lineage-receipt", type=Path, required=True)
    parser.add_argument("--candidate-summary", type=Path, required=True)
    parser.add_argument("--inference", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args(argv)
    outputs = (args.inference, args.labels, args.audit, args.receipt)
    if len({path.resolve() for path in outputs}) != len(outputs):
        parser.error("all output paths must be distinct")
    if any(path.exists() or path.is_symlink() for path in outputs):
        parser.error("output already exists")
    lineage = validate_trajectory_lineage_receipt(args.lineage_receipt)
    rows = load_jsonl(args.source)
    inference, labels, audit = split_rows(rows)
    bind_trajectory_identity(rows, inference, labels, audit)
    candidate_rows = load_jsonl(args.candidate_summary)
    candidates = {str(row.get("episode_id") or ""): row for row in candidate_rows}
    if "" in candidates or len(candidates) != len(candidate_rows):
        parser.error("candidate summary has missing/duplicate episode IDs")
    if set(candidates) != {str(row["episode_id"]) for row in inference}:
        parser.error("candidate summary does not exactly cover trajectory episodes")
    for row in inference:
        candidate = candidates[str(row["episode_id"])]
        row["candidate_json"] = str(candidate["candidate_path"])
        row["candidate_sha256"] = str(candidate["candidate_sha256"])
        _assert_visible_value_safe(row)
    label_map = {str(row["episode_id"]): row for row in labels}
    split_sha = validate_training_split(label_map, args.learning_split)
    _write_jsonl_exclusive(args.inference, inference)
    _write_jsonl_exclusive(args.labels, labels)
    _write_jsonl_exclusive(args.audit, audit)
    receipt = {
        "schema_version": TRAJECTORY_PROGRAM_RECEIPT_SCHEMA,
        "status": "PASS",
        "row_count": len(inference),
        "learning_split": {
            "path": str(args.learning_split.resolve()),
            "sha256": split_sha,
        },
        "source": {
            "path": str(args.source.resolve()),
            "sha256": _sha256_file(args.source),
        },
        "outputs": {
            "inference": file_record(args.inference),
            "labels": file_record(args.labels),
            "audit": file_record(args.audit),
        },
        "locked_test_present": False,
        "dataset_id": TRAJECTORY_DATASET_ID,
        "source_m0_lineage": MAINLINE_SOURCE,
        "mainline_eligible": False,
        "lifecycle": "active",
        "lineage_receipt": file_record(Path(lineage["receipt_path"])),
        "candidate_summary": file_record(args.candidate_summary),
    }
    receipt["binding_sha256"] = _canonical_sha(receipt)
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    with args.receipt.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(
            receipt, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False
        )
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({"status": "PASS", "rows": len(inference), "receipt": str(args.receipt)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
