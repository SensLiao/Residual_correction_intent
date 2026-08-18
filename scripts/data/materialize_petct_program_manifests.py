#!/usr/bin/env python3
"""Physically separate v3 inference, label, and audit manifests.

The input is the rich construction/tensor manifest.  The output inference
manifest is a strict allowlisted capability: it contains no patient/case,
goal, matched-group, evaluation path/hash, GT path, or split hash.  Training
and evaluation explicitly join the label manifest only after the visible
artifact is materialized.  The audit manifest retains the full provenance
outside the inference process.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from common.petct_program_learning import (  # noqa: E402
    LearningContractError,
    _sha256_file,
    load_jsonl,
    validate_training_split,
)
from common.petct_mainline_lineage import (  # noqa: E402
    MAINLINE_DATASET_ID,
    MAINLINE_SOURCE,
    file_record,
    validate_r13_lineage_receipt,
    validate_r13_program_rows,
)

INFERENCE_SCHEMA = "PETCT-PROGRAM-INFERENCE-MANIFEST-v1.0"
LABEL_SCHEMA = "PETCT-PROGRAM-LABEL-MANIFEST-v1.0"
AUDIT_SCHEMA = "PETCT-PROGRAM-AUDIT-MANIFEST-v1.0"
RECEIPT_SCHEMA = "PETCT-PROGRAM-MANIFEST-READY-v1.0"

FROZEN_GOALS = {
    "ADD_SAME_LOCAL",
    "ADD_SAME_COMPLETE",
    "ADD_NEW_COMPLETE",
    "REMOVE_SAME_LOCAL",
    "REMOVE_SAME_COMPLETE",
    "REMOVE_NEW_COMPLETE",
}


def _canonical_sha(payload: Any) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _assert_visible_value_safe(value: Any, path: str = "row") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            lowered = str(key).casefold()
            if any(token in lowered for token in (
                "goal", "gold", "label", "patient", "case_id", "group",
                "evaluation", "target", "authorized", "ground_truth", "gt_",
            )):
                raise LearningContractError(
                    "inference manifest contains forbidden key at %s.%s" % (path, key)
                )
            _assert_visible_value_safe(child, "%s.%s" % (path, key))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _assert_visible_value_safe(child, "%s[%d]" % (path, index))
    elif isinstance(value, str):
        normalized = value.replace("\\", "/").casefold()
        if value.upper() in FROZEN_GOALS or any(
            goal.casefold() in normalized for goal in FROZEN_GOALS
        ):
            raise LearningContractError(
                "inference manifest contains a label-derived value at %s" % path
            )


def split_rows(rows: Sequence[Mapping[str, Any]]) -> tuple[list[dict], list[dict], list[dict]]:
    inference_rows: list[dict] = []
    label_rows: list[dict] = []
    audit_rows: list[dict] = []
    seen: set[str] = set()
    for source in rows:
        episode_id = str(source.get("episode_id") or "")
        if not episode_id or episode_id in seen:
            raise LearningContractError("source manifest has missing/duplicate episode_id")
        seen.add(episode_id)
        if source.get("partition") not in ("train", "val"):
            raise LearningContractError(
                "program manifest materializer refuses locked test or unknown partitions"
            )
        required = {
            "case_id", "patient_id", "partition", "goal", "operation",
            "visible_npz", "visible_sha256",
            "evaluation_npz", "evaluation_sha256", "learning_split_sha256",
        }
        # Natural single-round rows carry no matched-state grouping; the R13
        # identity attaches episode_family_id in bind_r13_identity.  A
        # controlled-lane row that lost its group id fails closed here.
        has_matched_group = (
            "matched_state_group_id" in source
            and str(source.get("matched_state_group_id") or "")
        )
        if "matched_state_group_id" in source:
            required.add("matched_state_group_id")
        if not required.issubset(source):
            raise LearningContractError(
                "rich source row is incomplete for episode %s" % episode_id
            )
        inference = {
            "schema_version": INFERENCE_SCHEMA,
            "episode_id": episode_id,
            "partition": str(source["partition"]),
            "operation": str(source["operation"]),
            "visible_npz": str(source["visible_npz"]),
            "visible_sha256": str(source["visible_sha256"]),
        }
        for optional in ("geometry", "center_z"):
            if optional in source:
                inference[optional] = source[optional]
        _assert_visible_value_safe(inference)
        label = {
            "schema_version": LABEL_SCHEMA,
            "episode_id": episode_id,
            "case_id": str(source["case_id"]),
            "patient_id": str(source["patient_id"]),
            "partition": str(source["partition"]),
            "goal": str(source["goal"]),
            "operation": str(source["operation"]),
            "evaluation_npz": str(source["evaluation_npz"]),
            "evaluation_sha256": str(source["evaluation_sha256"]),
            "learning_split_sha256": str(source["learning_split_sha256"]),
        }
        if has_matched_group:
            label["matched_state_group_id"] = str(source["matched_state_group_id"])
        for optional in ("target", "scope", "held_out_fold", "test_access_receipt_sha256"):
            if optional in source:
                label[optional] = source[optional]
        if label["goal"] not in FROZEN_GOALS:
            raise LearningContractError("unknown frozen goal in label manifest")
        inference_rows.append(inference)
        label_rows.append(label)
        audit_rows.append(
            {
                "schema_version": AUDIT_SCHEMA,
                "episode_id": episode_id,
                "source_record_sha256": _canonical_sha(source),
                "source_record": dict(source),
            }
        )
    return inference_rows, label_rows, audit_rows


def bind_r13_identity(
    source_rows: Sequence[Mapping[str, Any]],
    inference_rows: list[dict],
    label_rows: list[dict],
    audit_rows: list[dict],
) -> None:
    """Attach the single-round R13 identity without exposing label identity."""

    for source, inference, label, audit in zip(
        source_rows, inference_rows, label_rows, audit_rows
    ):
        round_index = int(source.get("round_index", 0))
        strategy = str(source.get("strategy") or "")
        source_evaluation = source.get("source_evaluation")
        m0_sha256 = (
            str(source_evaluation.get("m0_sha256") or "")
            if isinstance(source_evaluation, Mapping)
            else ""
        )
        if not m0_sha256:
            raise LearningContractError("R13 source row lacks current-M content identity")
        family_id = hashlib.sha256(
            (
                "PETCT-R13-FAMILY-v1|%s|%s|%d|%s"
                % (
                    source["case_id"],
                    m0_sha256,
                    round_index,
                    source["operation"],
                )
            ).encode("utf-8")
        ).hexdigest()[:24]
        inference.update(
            {
                "dataset_id": MAINLINE_DATASET_ID,
                "source_m0_lineage": MAINLINE_SOURCE,
                "round_index": round_index,
                "scribble_count": 1,
            }
        )
        label.update(
            {
                "dataset_id": MAINLINE_DATASET_ID,
                "source_m0_lineage": MAINLINE_SOURCE,
                "episode_family_id": family_id,
                "round_index": round_index,
                "scribble_count": 1,
                "strategy": strategy,
            }
        )
        audit.update(
            {
                "dataset_id": MAINLINE_DATASET_ID,
                "source_m0_lineage": MAINLINE_SOURCE,
                "episode_family_id": family_id,
                "round_index": round_index,
                "scribble_count": 1,
                "strategy": strategy,
            }
        )
    validate_r13_program_rows(label_rows)


def _write_jsonl_exclusive(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(
                row, ensure_ascii=False, sort_keys=True, allow_nan=False
            ))
            stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


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
    lineage = validate_r13_lineage_receipt(args.lineage_receipt)
    rows = load_jsonl(args.source)
    inference, labels, audit = split_rows(rows)
    bind_r13_identity(rows, inference, labels, audit)
    candidate_rows = load_jsonl(args.candidate_summary)
    candidates = {str(row.get("episode_id") or ""): row for row in candidate_rows}
    if "" in candidates or len(candidates) != len(candidate_rows):
        parser.error("candidate summary has missing/duplicate episode IDs")
    if set(candidates) != {str(row["episode_id"]) for row in inference}:
        parser.error("candidate summary does not exactly cover R13 episodes")
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
        "schema_version": RECEIPT_SCHEMA,
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
        "dataset_id": MAINLINE_DATASET_ID,
        "source_m0_lineage": MAINLINE_SOURCE,
        "mainline_eligible": True,
        "lifecycle": "active",
        "lineage_receipt": file_record(Path(lineage["receipt_path"])),
        "candidate_summary": file_record(args.candidate_summary),
    }
    receipt["binding_sha256"] = _canonical_sha(receipt)
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    with args.receipt.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(receipt, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({"status": "PASS", "rows": len(inference), "receipt": str(args.receipt)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
