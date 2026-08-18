#!/usr/bin/env python3
"""Fail-closed lineage contract for the R13 five-round trajectory corpus.

The trajectory corpus (R13-trajectory-5r) shares the frozen M0 v6 OOF source
with R13-main but is an explicitly separate dataset: five teacher-forced
rounds per trajectory, ``mainline_eligible=False`` so no mainline scanner can
pick it up as the active single-round corpus.  This module never modifies the
single-round lineage module; it reuses its validation primitives.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from common.petct_mainline_lineage import (  # noqa: E402
    ALLOWED_PARTITIONS,
    ALLOWED_STRATEGIES,
    MAINLINE_SOURCE,
    LineageContractError,
    _load_json,
    _load_jsonl,
    _sha256_file,
    _validate_record,
    file_record,
    validate_m0_v6_oof_ready,
)

TRAJECTORY_DATASET_ID = "R13-trajectory-5r"
TRAJECTORY_LINEAGE_SCHEMA = "PETCT-R13-TRAJECTORY-LINEAGE-v1.0"
TRAJECTORY_DATA_READY_SCHEMA = "PETCT-R13-TRAJECTORY-DATA-READY-v1.0"
TRAJECTORY_PROGRAM_RECEIPT_SCHEMA = "PETCT-TRAJECTORY-PROGRAM-MANIFEST-READY-v1.0"
TRAJECTORY_EPISODE_SCHEMA = "five_round_trajectory_v1"
TRAJECTORY_ROUND_LIMIT = 5
TRAJECTORY_STATUSES = ("COMPLETE_5_ROUNDS", "RESIDUAL_EXHAUSTED", "TRUNCATED")


class TrajectoryContractError(LineageContractError):
    """Raised when an artifact violates the five-round trajectory contract."""


def validate_r13_trajectory_rows(rows: Iterable[Mapping[str, Any]]) -> None:
    """Enforce the trajectory sibling/round structure on program label rows.

    One trajectory is one (case, operation, strategy) chain of contiguous
    rounds starting at 0, at most five rounds long, with the same three-lane
    identity per row as R13-main and at most three strategy siblings per
    (case, operation, round_index).
    """

    seen: set[str] = set()
    trajectories: defaultdict[str, list[Mapping[str, Any]]] = defaultdict(list)
    count = 0
    for row in rows:
        count += 1
        episode_id = str(row.get("episode_id") or "")
        trajectory_id = str(row.get("trajectory_id") or "")
        family_id = str(row.get("episode_family_id") or "")
        if not episode_id or episode_id in seen:
            raise TrajectoryContractError(
                "R13 trajectory episode IDs must be non-empty and unique"
            )
        seen.add(episode_id)
        if not trajectory_id:
            raise TrajectoryContractError("R13 trajectory_id is missing")
        if family_id != trajectory_id:
            raise TrajectoryContractError(
                "episode_family_id must equal the trajectory_id"
            )
        if row.get("source_m0_lineage") != MAINLINE_SOURCE:
            raise TrajectoryContractError(
                "R13 trajectory source_m0_lineage must bind %s" % MAINLINE_SOURCE
            )
        if row.get("partition") not in ALLOWED_PARTITIONS:
            raise TrajectoryContractError(
                "R13 trajectory partition must be train or val"
            )
        round_index = row.get("round_index")
        if (
            isinstance(round_index, bool)
            or not isinstance(round_index, int)
            or round_index not in range(TRAJECTORY_ROUND_LIMIT)
        ):
            raise TrajectoryContractError(
                "round_index must be an integer in 0..4"
            )
        if row.get("scribble_count") != 1:
            raise TrajectoryContractError(
                "R13 trajectory requires exactly one scribble per episode"
            )
        if str(row.get("strategy") or "") not in ALLOWED_STRATEGIES:
            raise TrajectoryContractError(
                "R13 trajectory strategy is not a frozen simulator strategy"
            )
        trajectories[trajectory_id].append(row)
    if count == 0:
        raise TrajectoryContractError("R13 trajectory manifest is empty")
    for trajectory_id, members in trajectories.items():
        first = members[0]
        identity = {
            str(first.get(key) or "")
            for key in (
                "case_id",
                "patient_id",
                "operation",
                "strategy",
                "round_count",
                "trajectory_status",
            )
        }
        for member in members[1:]:
            candidate = {
                str(member.get(key) or "")
                for key in (
                    "case_id",
                    "patient_id",
                    "operation",
                    "strategy",
                    "round_count",
                    "trajectory_status",
                )
            }
            if candidate != identity:
                raise TrajectoryContractError(
                    "rows of one trajectory must share case, patient, "
                    "operation, strategy, round_count and trajectory_status"
                )
        round_count = members[0].get("round_count")
        if (
            isinstance(round_count, bool)
            or not isinstance(round_count, int)
            or not 1 <= round_count <= TRAJECTORY_ROUND_LIMIT
        ):
            raise TrajectoryContractError("round_count must be an integer in 1..5")
        rounds = sorted(int(member["round_index"]) for member in members)
        if rounds != list(range(len(members))):
            raise TrajectoryContractError(
                "trajectory rounds must be contiguous starting at 0"
            )
        if int(round_count) != len(members):
            raise TrajectoryContractError(
                "round_count must equal the number of trajectory rows"
            )
        if members[0].get("trajectory_status") not in TRAJECTORY_STATUSES:
            raise TrajectoryContractError(
                "trajectory_status must be one of the three frozen statuses"
            )
    siblings: defaultdict[tuple[str, str, int], list[str]] = defaultdict(list)
    for row in rows:
        key = (
            str(row["case_id"]),
            str(row["operation"]),
            int(row["round_index"]),
        )
        siblings[key].append(str(row["strategy"]))
    for key, strategies in siblings.items():
        if len(strategies) > 3:
            raise TrajectoryContractError(
                "R13 trajectory round %s has at most three strategy siblings" % (key,)
            )
        if len(strategies) != len(set(strategies)):
            raise TrajectoryContractError(
                "R13 trajectory round %s repeats a strategy sibling" % (key,)
            )


def validate_trajectory_lineage_receipt(path: Path) -> dict[str, Any]:
    path = path.resolve()
    receipt = _load_json(path, label="R13 trajectory lineage receipt")
    required = {
        "schema_version": TRAJECTORY_LINEAGE_SCHEMA,
        "status": "PASS",
        "dataset_id": TRAJECTORY_DATASET_ID,
        "source_m0_lineage": MAINLINE_SOURCE,
        "mainline_eligible": False,
        "lifecycle": "active",
        "episode_schema": TRAJECTORY_EPISODE_SCHEMA,
        "round_count": TRAJECTORY_ROUND_LIMIT,
        "scribbles_per_episode": 1,
        "strategy_is_label": False,
        "partitions": ["train", "val"],
        "locked_test_present": False,
        "teacher_forcing": "ORACLE_AUTHORIZED_TARGET",
    }
    for key, expected in required.items():
        if receipt.get(key) != expected:
            raise TrajectoryContractError(
                "trajectory lineage field %s is invalid" % key
            )
    for key in ("oof_ready", "learning_split", "experiment_config"):
        _validate_record(receipt.get(key), label="trajectory %s" % key)
    return {
        **receipt,
        "receipt_path": str(path),
        "receipt_sha256": _sha256_file(path),
    }


def issue_trajectory_lineage_receipt(
    *,
    oof_ready: Path,
    learning_split: Path,
    experiment_config: Path,
    output: Path,
) -> dict[str, Any]:
    if output.exists() or output.is_symlink():
        raise TrajectoryContractError(
            "refusing to overwrite R13 trajectory lineage receipt"
        )
    try:
        validate_m0_v6_oof_ready(oof_ready)
    except LineageContractError as exc:
        raise TrajectoryContractError(
            "trajectory lineage OOF gate failed: %s" % exc
        ) from exc
    payload = {
        "schema_version": TRAJECTORY_LINEAGE_SCHEMA,
        "status": "PASS",
        "dataset_id": TRAJECTORY_DATASET_ID,
        "source_m0_lineage": MAINLINE_SOURCE,
        "mainline_eligible": False,
        "lifecycle": "active",
        "episode_schema": TRAJECTORY_EPISODE_SCHEMA,
        "round_count": TRAJECTORY_ROUND_LIMIT,
        "scribbles_per_episode": 1,
        "strategy_is_label": False,
        "partitions": ["train", "val"],
        "locked_test_present": False,
        "teacher_forcing": "ORACLE_AUTHORIZED_TARGET",
        "oof_ready": file_record(oof_ready),
        "learning_split": file_record(learning_split),
        "experiment_config": file_record(experiment_config),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    validate_trajectory_lineage_receipt(output)
    return payload


def validate_trajectory_program_receipt(
    receipt_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    receipt = _load_json(receipt_path, label="trajectory program manifest receipt")
    required = {
        "schema_version": TRAJECTORY_PROGRAM_RECEIPT_SCHEMA,
        "status": "PASS",
        "dataset_id": TRAJECTORY_DATASET_ID,
        "source_m0_lineage": MAINLINE_SOURCE,
        "mainline_eligible": False,
        "lifecycle": "active",
        "locked_test_present": False,
    }
    for key, expected in required.items():
        if receipt.get(key) != expected:
            raise TrajectoryContractError(
                "trajectory program receipt field %s is invalid" % key
            )
    bound_lineage = receipt.get("lineage_receipt")
    if not isinstance(bound_lineage, Mapping):
        raise TrajectoryContractError(
            "trajectory program receipt lacks lineage binding"
        )
    lineage_path = _validate_record(bound_lineage, label="bound trajectory lineage")
    return receipt, {"path": lineage_path}


def validate_trajectory_training_binding(
    lineage_receipt: Path,
    manifest_receipt: Path,
    episodes: Path,
    labels: Path,
) -> dict[str, Any]:
    lineage = validate_trajectory_lineage_receipt(lineage_receipt)
    manifest, bound = validate_trajectory_program_receipt(manifest_receipt)
    bound_lineage = manifest.get("lineage_receipt")
    if (
        Path(bound["path"]) != Path(lineage["receipt_path"])
        or bound_lineage.get("sha256") != lineage["receipt_sha256"]
    ):
        raise TrajectoryContractError(
            "trajectory program manifest binds a different lineage receipt"
        )
    outputs = manifest.get("outputs")
    if not isinstance(outputs, Mapping):
        raise TrajectoryContractError("trajectory program outputs are missing")
    for name, supplied in (("inference", episodes), ("labels", labels)):
        bound_record = _validate_record(
            outputs.get(name), label="trajectory program %s" % name
        )
        if bound_record != supplied.resolve():
            raise TrajectoryContractError(
                "trajectory program %s path differs from receipt" % name
            )
    validate_r13_trajectory_rows(_load_jsonl(labels, label="R13 trajectory labels"))
    return lineage


def seal_trajectory_data_ready(
    *,
    lineage_receipt: Path,
    manifest_receipt: Path,
    inference_manifest: Path,
    label_manifest: Path,
    audit_manifest: Path,
    rich_tensor_manifest: Path,
    candidate_summary: Path,
    pointer_summary: Path,
    trajectories_summary: Path,
    output: Path,
) -> dict[str, Any]:
    lineage = validate_trajectory_training_binding(
        lineage_receipt, manifest_receipt, inference_manifest, label_manifest
    )
    if output.exists() or output.is_symlink():
        raise TrajectoryContractError(
            "refusing to overwrite R13 trajectory data-ready receipt"
        )
    payload = {
        "schema_version": TRAJECTORY_DATA_READY_SCHEMA,
        "status": "PASS",
        "dataset_id": TRAJECTORY_DATASET_ID,
        "source_m0_lineage": MAINLINE_SOURCE,
        "mainline_eligible": False,
        "lifecycle": "active",
        "locked_test_present": False,
        "lineage_receipt": file_record(lineage_receipt),
        "program_manifest_receipt": file_record(manifest_receipt),
        "outputs": {
            "inference_visible": file_record(inference_manifest),
            "label_only": file_record(label_manifest),
            "audit_only": file_record(audit_manifest),
            "rich_tensors": file_record(rich_tensor_manifest),
            "candidate_summary": file_record(candidate_summary),
            "pointer_summary": file_record(pointer_summary),
            "trajectories_summary": file_record(trajectories_summary),
        },
        "row_count": len(
            _load_jsonl(label_manifest, label="R13 trajectory label manifest")
        ),
        "lineage_receipt_sha256": lineage["receipt_sha256"],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    return payload


def validate_trajectory_data_ready(path: Path) -> dict[str, Any]:
    path = path.resolve()
    ready = _load_json(path, label="R13 trajectory data-ready receipt")
    required = {
        "schema_version": TRAJECTORY_DATA_READY_SCHEMA,
        "status": "PASS",
        "dataset_id": TRAJECTORY_DATASET_ID,
        "source_m0_lineage": MAINLINE_SOURCE,
        "mainline_eligible": False,
        "lifecycle": "active",
        "locked_test_present": False,
    }
    for key, expected in required.items():
        if ready.get(key) != expected:
            raise TrajectoryContractError(
                "trajectory data-ready field %s is invalid" % key
            )
    lineage_path = _validate_record(
        ready.get("lineage_receipt"), label="trajectory lineage receipt"
    )
    lineage = validate_trajectory_lineage_receipt(lineage_path)
    _validate_record(
        ready.get("program_manifest_receipt"),
        label="trajectory program manifest receipt",
    )
    outputs = ready.get("outputs")
    if not isinstance(outputs, Mapping):
        raise TrajectoryContractError("trajectory data-ready outputs are missing")
    for key in (
        "inference_visible",
        "label_only",
        "audit_only",
        "rich_tensors",
        "candidate_summary",
        "pointer_summary",
        "trajectories_summary",
    ):
        _validate_record(outputs.get(key), label="trajectory %s" % key)
    return {
        **ready,
        "receipt_path": str(path),
        "receipt_sha256": _sha256_file(path),
        "validated_lineage": lineage,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    issue = commands.add_parser("issue")
    issue.add_argument("--oof-ready", type=Path, required=True)
    issue.add_argument("--learning-split", type=Path, required=True)
    issue.add_argument("--experiment-config", type=Path, required=True)
    issue.add_argument("--output", type=Path, required=True)
    validate = commands.add_parser("validate")
    validate.add_argument("--receipt", type=Path, required=True)
    validate_data = commands.add_parser("validate-data")
    validate_data.add_argument("--receipt", type=Path, required=True)
    seal = commands.add_parser("seal")
    seal.add_argument("--lineage-receipt", type=Path, required=True)
    seal.add_argument("--manifest-receipt", type=Path, required=True)
    seal.add_argument("--inference-manifest", type=Path, required=True)
    seal.add_argument("--label-manifest", type=Path, required=True)
    seal.add_argument("--audit-manifest", type=Path, required=True)
    seal.add_argument("--rich-tensor-manifest", type=Path, required=True)
    seal.add_argument("--candidate-summary", type=Path, required=True)
    seal.add_argument("--pointer-summary", type=Path, required=True)
    seal.add_argument("--trajectories-summary", type=Path, required=True)
    seal.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "issue":
        payload = issue_trajectory_lineage_receipt(
            oof_ready=args.oof_ready,
            learning_split=args.learning_split,
            experiment_config=args.experiment_config,
            output=args.output,
        )
    elif args.command == "validate":
        payload = validate_trajectory_lineage_receipt(args.receipt)
    elif args.command == "validate-data":
        payload = validate_trajectory_data_ready(args.receipt)
    else:
        payload = seal_trajectory_data_ready(
            lineage_receipt=args.lineage_receipt,
            manifest_receipt=args.manifest_receipt,
            inference_manifest=args.inference_manifest,
            label_manifest=args.label_manifest,
            audit_manifest=args.audit_manifest,
            rich_tensor_manifest=args.rich_tensor_manifest,
            candidate_summary=args.candidate_summary,
            pointer_summary=args.pointer_summary,
            trajectories_summary=args.trajectories_summary,
            output=args.output,
        )
    print(json.dumps({"status": "PASS", "dataset_id": payload["dataset_id"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
