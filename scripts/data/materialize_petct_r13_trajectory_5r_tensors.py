#!/usr/bin/env python3
"""Five-round variant of the 2.5D tensor materializer.

Reuses the frozen ``materialize_episode`` unchanged (same normalization,
crop, and physical-lane split as R13-main).  The only five-round behavior is
validating the explicit trajectory identity of each row and copying it onto
the materialized rows; every tensor itself is an ordinary single-round
episode tensor whose current-M channel encodes the teacher-forced state.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Sequence

import numpy as np

SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from common.petct_learning import (  # noqa: E402
    LearningContractError,
    load_jsonl,
    sha256_file,
    validate_manifest_rows_against_frozen_learning_split,
    validate_partitions,
)
from common.petct_test_access import (  # noqa: E402
    TestAccessError,
    add_leaf_test_access_arguments,
    enforce_partition_access,
)
from common.petct_trajectory_lineage import (  # noqa: E402
    TRAJECTORY_ROUND_LIMIT,
    TRAJECTORY_STATUSES,
)
from data.materialize_petct_learning_tensors import (  # noqa: E402
    materialize_episode,
    staged_output_bundle,
)


def _validate_trajectory_identity(
    row, seen_pairs: set[tuple[str, int]]
) -> None:
    trajectory_id = str(row.get("trajectory_id") or "")
    if not trajectory_id or Path(trajectory_id).name != trajectory_id:
        raise LearningContractError("row lacks a filename-safe trajectory_id")
    round_index = row.get("round_index")
    if (
        isinstance(round_index, bool)
        or not isinstance(round_index, int)
        or round_index not in range(TRAJECTORY_ROUND_LIMIT)
    ):
        raise LearningContractError("row has invalid round_index")
    round_count = row.get("round_count")
    if (
        isinstance(round_count, bool)
        or not isinstance(round_count, int)
        or not 1 <= round_count <= TRAJECTORY_ROUND_LIMIT
    ):
        raise LearningContractError("row has invalid round_count")
    if int(round_index) >= int(round_count):
        raise LearningContractError("row round_index exceeds round_count")
    if str(row.get("trajectory_status") or "") not in TRAJECTORY_STATUSES:
        raise LearningContractError("row has invalid trajectory_status")
    pair = (trajectory_id, int(round_index))
    if pair in seen_pairs:
        raise LearningContractError("duplicate (trajectory_id, round_index) pair")
    seen_pairs.add(pair)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-manifest", type=Path, required=True)
    parser.add_argument("--visible-root", type=Path, required=True)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--experiment-config", type=Path, required=True)
    parser.add_argument("--learning-split", type=Path, required=True)
    parser.add_argument(
        "--partitions",
        nargs="+",
        choices=("train", "val", "test"),
        required=True,
    )
    add_leaf_test_access_arguments(parser)
    args = parser.parse_args(argv)
    if len(args.partitions) != len(set(args.partitions)):
        parser.error("--partitions must not contain duplicates")
    roots = [path.resolve() for path in (args.visible_root, args.evaluation_root)]
    output_manifest = args.output_manifest.resolve()
    selected_partitions = set(args.partitions)
    try:
        test_access = enforce_partition_access(
            selected_partitions,
            receipt_path=args.test_access_receipt,
            experiment_config=args.experiment_config,
            learning_split=args.learning_split,
            run_root=args.run_root,
            output_paths=(*roots, output_manifest),
        )
    except TestAccessError as error:
        parser.error(str(error))
    test_access_sha256 = (
        None if test_access is None else str(test_access["receipt_sha256"])
    )
    with args.experiment_config.open("r", encoding="utf-8") as stream:
        experiment_config = json.load(stream)
    tensor_config = experiment_config["learning_tensor_normalization"]
    field_mm = float(tensor_config["crop_field_mm"])
    output_size = int(tensor_config["output_size_px"])
    expected_spacing = float(tensor_config["output_spacing_mm"])
    if not np.isclose(field_mm / output_size, expected_spacing, atol=1e-9, rtol=0):
        parser.error("output spacing does not equal crop_field_mm/output_size_px")
    config_sha256 = sha256_file(args.experiment_config)
    if roots[0] == roots[1] or roots[0] in roots[1].parents or roots[1] in roots[0].parents:
        parser.error("visible and evaluation roots must be physically disjoint")
    if any(os.path.lexists(str(path)) for path in (*roots, output_manifest)):
        parser.error("output paths must not already exist")
    source_rows = load_jsonl(args.episode_manifest)
    try:
        validate_manifest_rows_against_frozen_learning_split(
            source_rows,
            args.learning_split,
            require_episode_id=True,
            allowed_partitions=selected_partitions,
        )
    except LearningContractError as error:
        parser.error(str(error))
    seen_pairs: set[tuple[str, int]] = set()
    for row_number, row in enumerate(source_rows, start=1):
        expected_receipt = (
            test_access_sha256 if row.get("partition") == "test" else None
        )
        if row.get("test_access_receipt_sha256") != expected_receipt:
            parser.error(
                "episode row %d test-access receipt provenance mismatch" % row_number
            )
        try:
            _validate_trajectory_identity(row, seen_pairs)
        except LearningContractError as error:
            parser.error("trajectory row %d: %s" % (row_number, error))
    output_rows = []
    with staged_output_bundle(
        directory_outputs=roots, file_outputs=[output_manifest]
    ) as staged:
        for row in source_rows:
            materialized = materialize_episode(
                row,
                staged_visible_root=staged[roots[0]],
                staged_evaluation_root=staged[roots[1]],
                final_visible_root=roots[0],
                final_evaluation_root=roots[1],
                field_mm=field_mm,
                output_size=output_size,
                expected_spacing=expected_spacing,
                config_sha256=config_sha256,
            )
            materialized["trajectory_id"] = str(row["trajectory_id"])
            materialized["round_index"] = int(row["round_index"])
            materialized["round_count"] = int(row["round_count"])
            materialized["trajectory_status"] = str(row["trajectory_status"])
            if row.get("termination_reason") is not None:
                materialized["termination_reason"] = str(row["termination_reason"])
            output_rows.append(materialized)
        validate_partitions(output_rows)
        with staged[output_manifest].open(
            "x", encoding="utf-8", newline="\n"
        ) as stream:
            for row in output_rows:
                stream.write(
                    json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                )
    print(
        json.dumps(
            {
                "episodes": len(output_rows),
                "visible_root": str(args.visible_root),
                "evaluation_root": str(args.evaluation_root),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
