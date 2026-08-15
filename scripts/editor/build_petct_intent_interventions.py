#!/usr/bin/env python3
"""Freeze a deterministic, auditable non-identity shuffled-intent manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from common.petct_learning import (  # noqa: E402
    INTERVENTION_SCHEMA,
    LearningContractError,
    SHUFFLE_ALGORITHM,
    load_experiment_config,
    load_intent_intervention_contract,
    load_jsonl,
    sha256_file,
    validate_frozen_override,
    write_jsonl_exclusive,
    validate_manifest_rows_against_frozen_learning_split,
)
from common.petct_test_access import (  # noqa: E402
    TestAccessError,
    add_leaf_test_access_arguments,
    enforce_partition_access,
)


def _seeded_rank(seed: int, namespace: str, value: str) -> str:
    return hashlib.sha256(
        ("%s|%d|%s" % (namespace, seed, value)).encode("utf-8")
    ).hexdigest()


def build_nonidentity_shuffle(rows, *, seed: int):
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise RuntimeError("shuffle seed must be an integer")
    by_episode = {}
    groups = {}
    for raw in rows:
        episode = str(raw.get("episode_id") or "")
        joint = (
            str(raw.get("operation") or ""),
            str(raw.get("target") or ""),
            str(raw.get("scope") or ""),
        )
        if not episode:
            raise RuntimeError("shuffle rows require episode_id")
        if episode in by_episode:
            raise RuntimeError("duplicate shuffle episode_id: %s" % episode)
        if joint not in {
            (operation, target, scope)
            for operation in ("ADD", "REMOVE")
            for target, scope in (
                ("SAME", "LOCAL"),
                ("SAME", "COMPLETE"),
                ("NEW", "COMPLETE"),
            )
        }:
            raise RuntimeError("shuffle rows contain an illegal joint label")
        row = dict(raw)
        by_episode[episode] = row
        groups.setdefault(joint, []).append(row)
    if len(by_episode) < 2:
        raise RuntimeError("shuffle requires at least two episodes")

    # For a slot-changing permutation, the largest label stratum cannot exceed
    # the union of all other strata.  Failing closed is preferable to emitting
    # reused donors or silently unchanged interventions.
    largest = max(len(group) for group in groups.values())
    if largest > len(by_episode) - largest:
        raise RuntimeError(
            "joint-label-preserving derangement is infeasible: largest stratum "
            "has %d of %d episodes" % (largest, len(by_episode))
        )

    label_order = sorted(
        groups,
        key=lambda joint: (
            _seeded_rank(
                seed,
                "PETCT-SHUFFLE-LABEL-v2",
                "%s|%s|%s" % joint,
            ),
            joint,
        ),
    )
    ring = []
    for joint in label_order:
        ring.extend(
            sorted(
                groups[joint],
                key=lambda row: (
                    _seeded_rank(
                        seed,
                        "PETCT-SHUFFLE-EPISODE-v1",
                        str(row["episode_id"]),
                    ),
                    str(row["episode_id"]),
                ),
            )
        )
    donors = ring[largest:] + ring[:largest]
    output = []
    for row, source in zip(ring, donors):
        joint = (
            str(row["operation"]),
            str(row["target"]),
            str(row["scope"]),
        )
        source_joint = (
            str(source["operation"]),
            str(source["target"]),
            str(source["scope"]),
        )
        if (
            str(source["episode_id"]) == str(row["episode_id"])
            or source_joint == joint
        ):
            raise RuntimeError(
                "internal error: constructed shuffle is not a derangement"
            )
        output.append(
            {
                "schema_version": INTERVENTION_SCHEMA,
                "algorithm": SHUFFLE_ALGORITHM,
                "seed": int(seed),
                "permutation_size": len(by_episode),
                "episode_id": str(row["episode_id"]),
                "source_episode_id": str(source["episode_id"]),
                "operation": source_joint[0],
                "target": source_joint[1],
                "scope": source_joint[2],
                "original_operation": joint[0],
                "original_target": joint[1],
                "original_scope": joint[2],
                "changed": True,
            }
        )
    if len({row["source_episode_id"] for row in output}) != len(output):
        raise RuntimeError("internal error: shuffle reused a donor")
    original_counts = Counter(
        (
            row["original_operation"],
            row["original_target"],
            row["original_scope"],
        )
        for row in output
    )
    shuffled_counts = Counter(
        (row["operation"], row["target"], row["scope"]) for row in output
    )
    if original_counts != shuffled_counts:
        raise RuntimeError("internal error: shuffle changed joint-label marginals")
    return sorted(output, key=lambda row: row["episode_id"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--learning-manifest", type=Path, required=True)
    parser.add_argument("--learning-split", type=Path, required=True)
    parser.add_argument("--experiment-config", type=Path, required=True)
    parser.add_argument("--partition", choices=["val", "test"], required=True)
    parser.add_argument(
        "--seed",
        type=int,
        help="optional assertion against editor.intent_interventions.shuffle_seed",
    )
    parser.add_argument("--output", type=Path, required=True)
    add_leaf_test_access_arguments(parser)
    args = parser.parse_args()
    try:
        test_access = enforce_partition_access(
            args.partition,
            receipt_path=args.test_access_receipt,
            experiment_config=args.experiment_config,
            learning_split=args.learning_split,
            run_root=args.run_root,
            output_paths=(args.output,),
        )
    except TestAccessError as error:
        parser.error(str(error))
    test_access_sha256 = (
        None if test_access is None else str(test_access["receipt_sha256"])
    )
    if args.output.exists():
        parser.error("output already exists")
    try:
        config = load_experiment_config(args.experiment_config)
        intervention_contract = load_intent_intervention_contract(config)
        validate_frozen_override(
            "--seed", args.seed, intervention_contract["shuffle_seed"]
        )
    except LearningContractError as error:
        parser.error(str(error))
    seed = intervention_contract["shuffle_seed"]
    source_rows = load_jsonl(args.learning_manifest)
    try:
        validate_manifest_rows_against_frozen_learning_split(
            source_rows,
            args.learning_split,
            require_episode_id=True,
            allowed_partitions={"train", "val", "test"},
        )
    except LearningContractError as error:
        parser.error(str(error))
    rows = [row for row in source_rows if row["partition"] == args.partition]
    if not rows:
        parser.error("partition is empty")
    learning_split_sha256 = sha256_file(args.learning_split)
    if {row.get("learning_split_sha256") for row in rows} != {
        learning_split_sha256
    }:
        parser.error("learning manifest differs from the receipt-bound learning split")
    try:
        shuffled = build_nonidentity_shuffle(rows, seed=seed)
    except RuntimeError as error:
        parser.error(str(error))
    source_manifest_sha256 = sha256_file(args.learning_manifest)
    experiment_config_sha256 = sha256_file(args.experiment_config)
    source_by_episode = {str(row["episode_id"]): row for row in rows}
    for row in shuffled:
        donor = source_by_episode[row["source_episode_id"]]
        donor_visible_sha256 = str(donor.get("visible_sha256") or "")
        if not donor_visible_sha256:
            parser.error("learning manifest donor omits visible_sha256")
        row.update(
            {
                "partition": args.partition,
                "source_manifest_sha256": source_manifest_sha256,
                "experiment_config_sha256": experiment_config_sha256,
                "learning_split_sha256": learning_split_sha256,
                "test_access_receipt_sha256": test_access_sha256,
                "source_visible_npz_sha256": donor_visible_sha256,
            }
        )
    write_jsonl_exclusive(args.output, shuffled)
    print(
        json.dumps(
            {
                "episodes": len(shuffled),
                "partition": args.partition,
                "seed": seed,
                "algorithm": SHUFFLE_ALGORITHM,
                "schema_version": INTERVENTION_SCHEMA,
                "source_manifest_sha256": source_manifest_sha256,
                "experiment_config_sha256": experiment_config_sha256,
                "learning_split_sha256": learning_split_sha256,
                "test_access_receipt_sha256": test_access_sha256,
                "intervention_manifest_sha256": sha256_file(args.output),
                "all_interventions_changed": True,
                "donors_without_replacement": True,
                "joint_label_marginals_preserved": True,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
