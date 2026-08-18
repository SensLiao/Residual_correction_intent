#!/usr/bin/env python3
"""Render evaluator-only oracle calls for the natural single-round lane (R13-main).

The frozen ``render_petct_gold_program_calls_v3.py`` consumes the controlled
matched-state lane and therefore requires ``matched_state_group_id`` triplets.
R13-main natural labels (single round, strategy siblings) carry no matched
groups, so this natural-lane variant reuses the identical contract helpers and
derive loop with ``require_matched_groups=False``.  It is the only builder the
R13 effect-val gold-call ceiling pass may consume.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Sequence

SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from common.petct_program_contract import (  # noqa: E402
    GOAL_TO_FAMILY,
    NEW_CUE_SENTINEL,
    protected_refs_policy,
    render_goal,
    validate_legal_call,
)
from common.petct_program_learning import (  # noqa: E402
    LearningContractError,
    _sha256_file,
    load_label_manifest,
)

ORACLE_SCHEMA = "PETCT-PROGRAM-ORACLE-CALLS-v1.0"
ORACLE_RECEIPT_SCHEMA = "PETCT-PROGRAM-ORACLE-CALLS-READY-v1.0"


def _load_sidecars(directory: Path):
    if not directory.is_dir():
        raise LearningContractError("missing oracle sidecar directory")
    result = {}
    for path in sorted(directory.glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        episode = str(record.get("episode_id") or "")
        if not episode or episode in result:
            raise LearningContractError("oracle sidecars have missing/duplicate episode_id")
        result[episode] = record
    return result


def _tree_sha(directory: Path) -> str:
    rows = [
        (path.relative_to(directory).as_posix(), _sha256_file(path))
        for path in sorted(value for value in directory.rglob("*") if value.is_file())
    ]
    return hashlib.sha256(
        json.dumps(rows, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--pointer-targets", type=Path, required=True)
    parser.add_argument("--partition", choices=("train", "val"), default="val")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args(argv)
    if any(path.exists() or path.is_symlink() for path in (args.output, args.receipt)):
        parser.error("output already exists")
    labels = load_label_manifest(args.labels, require_matched_groups=False)
    candidates = _load_sidecars(args.candidates)
    targets = _load_sidecars(args.pointer_targets)
    calls = []
    for episode_id, label in labels.items():
        if label["partition"] != args.partition:
            continue
        candidate = candidates.get(episode_id)
        if candidate is None:
            raise LearningContractError("oracle call lacks candidate record")
        family = GOAL_TO_FAMILY[str(label["goal"])]
        operation = str(label["operation"])
        components = candidate.get("components", [])
        if family == "CREATE_NEW":
            operand = NEW_CUE_SENTINEL
            selection = "NEW_CUE"
        elif operation == "REMOVE":
            position = candidate.get("cue_hit_component_position")
            if position is None or not 0 <= int(position) < len(components):
                raise LearningContractError("oracle REMOVE lacks cue-hit component")
            operand = str(components[int(position)]["component_key"])
            selection = "deterministic_cue_hit"
        else:
            target = targets.get(episode_id)
            if target is None or not target.get("pointer_target_positions"):
                raise LearningContractError("oracle ADD existing lacks pointer positives")
            positive = [int(value) for value in target["pointer_target_positions"]]
            position = min(
                positive,
                key=lambda value: (
                    float(components[value]["distance_from_cue_mm"]), value
                ),
            )
            operand = str(components[position]["component_key"])
            selection = "nearest_gold_positive_then_position"
        validate_legal_call(operation, family, operand)
        calls.append(
            {
                "schema_version": ORACLE_SCHEMA,
                "episode_id": episode_id,
                "decision": "PREDICT",
                "operation": operation,
                "family": family,
                "operand": operand,
                "goal": render_goal(operation, family),
                "protected_refs": dict(protected_refs_policy(operation, operand)),
                "oracle_selection": selection,
                "source_lane": "evaluator_label_only",
            }
        )
    if not calls:
        raise LearningContractError("oracle partition is empty")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        for call in calls:
            stream.write(json.dumps(call, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    receipt = {
        "schema_version": ORACLE_RECEIPT_SCHEMA,
        "status": "PASS",
        "partition": args.partition,
        "call_count": len(calls),
        "calls_path": str(args.output.resolve()),
        "calls_sha256": _sha256_file(args.output),
        "labels_sha256": _sha256_file(args.labels),
        "candidates_tree_sha256": _tree_sha(args.candidates),
        "pointer_targets_tree_sha256": _tree_sha(args.pointer_targets),
        "selection_policy": "nearest physical cue distance among gold-positive operands; stable position tie-break",
        "source_lane": "evaluator_label_only",
        "label_lane_opened": True,
        "oracle_calls": True,
        "thesis_result": False,
        "natural_lane": True,
        "matched_groups_required": False,
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    with args.receipt.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({"status": "PASS", "oracle_calls": len(calls)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
