#!/usr/bin/env python3
"""Build the five-round teacher-forced trajectory corpus (R13-trajectory-5r).

Round 0 of every trajectory IS the frozen R13-main single-round episode: same
attempt id, episode id, visible/evaluation documents and generation receipt.
Rounds 1-4 advance the current state by the oracle authorized target of the
previous round (ADD = union, REMOVE = difference), draw the next residual
scribble with the pinned simulator, and derive the next gold goal with the
identical state-relative derivation the single-round builder uses.  Only the
round count differs from R13-main: strategy geometry, sibling structure,
exclusion rules and the three-lane split are unchanged.

The pure trajectory primitives live in ``petct_trajectory_primitives`` and
the per-trajectory round loop in ``build_petct_trajectory_round``; this
module owns argument parsing, frozen-input validation, and the staged
publication pipeline.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import nibabel as nib
import numpy as np

SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))
for support_dir in (SCRIPTS_ROOT / "baseline", SCRIPTS_ROOT / "data"):
    if str(support_dir) not in sys.path:
        sys.path.insert(0, str(support_dir))

from baseline.validate_petct_m0_oof import (  # noqa: E402
    build_natural_oof_binding_from_validated,
)
from common.petct_learning import (  # noqa: E402
    LearningContractError,
    load_jsonl,
    sha256_file,
    validate_manifest_rows_against_frozen_learning_split,
)
from common.petct_mainline_lineage import (  # noqa: E402
    validate_m0_v6_oof_ready,
)
from common.petct_test_access import (  # noqa: E402
    TestAccessError,
    add_leaf_test_access_arguments,
    enforce_partition_access,
)
from data.build_petct_scribble_dataset import (  # noqa: E402
    _canonical_hash,
    _cohort_bucket,
    _exclusion_summary,
    _output_file_record,
    _tree_record,
    _verified_source_path,
    resolve_scribble_generation_contract,
    scribble_attempt_id,
    selected_strategies,
    staged_output_bundle,
    validate_official_simulator_provenance,
    validate_residual_ready,
    verify_full_residual,
)
from data.build_petct_scribble_episode import (  # noqa: E402
    DEFAULT_RUNTIME_MANIFEST,
    load_official_simulator,
)
from data.build_petct_trajectory_round import _build_trajectory  # noqa: E402
from data.petct_trajectory_primitives import (  # noqa: E402
    PARITY_SINGLE_ROUND_DATASET,
    STATUS_COMPLETE,
    STATUS_EXHAUSTED,
    STATUS_TRUNCATED,
    TRAJECTORY_EVAL_SCHEMA,
    TRAJECTORY_READY_PHASE,
    TRAJECTORY_READY_SCHEMA,
    TRAJECTORY_ROW_KEYS,
    TRAJECTORY_STATE_SCHEMA,
    TRAJECTORY_VISIBLE_SCHEMA,
    _single_round_projection,
    advance_trajectory_state,
    build_state_provenance,
    build_trajectory_round_documents,
    round0_episode_id,
    teacher_forced_state,
    trajectory_attempt_id,
    trajectory_episode_id,
    trajectory_id,
)
from data.petct_trajectory_primitives import EpisodeContractError  # noqa: E402, F401

# Re-exported surface for the frozen test imports.
__all__ = (
    "EpisodeContractError",
    "PARITY_SINGLE_ROUND_DATASET",
    "STATUS_COMPLETE",
    "STATUS_EXHAUSTED",
    "STATUS_TRUNCATED",
    "TRAJECTORY_EVAL_SCHEMA",
    "TRAJECTORY_READY_PHASE",
    "TRAJECTORY_READY_SCHEMA",
    "TRAJECTORY_ROW_KEYS",
    "TRAJECTORY_STATE_SCHEMA",
    "TRAJECTORY_VISIBLE_SCHEMA",
    "advance_trajectory_state",
    "build_state_provenance",
    "build_trajectory_round_documents",
    "round0_episode_id",
    "teacher_forced_state",
    "trajectory_attempt_id",
    "trajectory_episode_id",
    "trajectory_id",
)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--residual-manifest", type=Path, required=True)
    parser.add_argument("--residual-ready", type=Path, required=True)
    parser.add_argument("--official-simulator", type=Path, required=True)
    parser.add_argument(
        "--official-runtime-manifest",
        type=Path,
        default=DEFAULT_RUNTIME_MANIFEST,
    )
    parser.add_argument("--experiment-config", type=Path, required=True)
    parser.add_argument("--learning-split", type=Path, required=True)
    parser.add_argument(
        "--partitions",
        nargs="+",
        choices=("train", "val", "test"),
        required=True,
    )
    parser.add_argument("--official-commit")
    parser.add_argument("--strategy-mode", choices=["primary", "all"])
    parser.add_argument("--strategy-salt")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--lane", choices=["natural"], default="natural")
    parser.add_argument("--oof-ready", type=Path, required=True)
    parser.add_argument("--visible-root", type=Path, required=True)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--authorized-root", type=Path, required=True)
    parser.add_argument("--state-root", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--trajectories", type=Path, required=True)
    parser.add_argument("--exclusions", type=Path, required=True)
    parser.add_argument("--ready-receipt", type=Path, required=True)
    add_leaf_test_access_arguments(parser)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if len(args.partitions) != len(set(args.partitions)):
        raise SystemExit("--partitions must not contain duplicates")
    final_directories = [
        args.visible_root.resolve(),
        args.evaluation_root.resolve(),
        args.authorized_root.resolve(),
        args.state_root.resolve(),
    ]
    final_files = [
        args.exclusions.resolve(),
        args.output_manifest.resolve(),
        args.trajectories.resolve(),
        args.ready_receipt.resolve(),
    ]
    selected_partitions = set(args.partitions)
    try:
        test_access = enforce_partition_access(
            selected_partitions,
            receipt_path=args.test_access_receipt,
            experiment_config=args.experiment_config,
            learning_split=args.learning_split,
            run_root=args.run_root,
            output_paths=(*final_directories, *final_files),
        )
    except TestAccessError as error:
        raise SystemExit(str(error))
    test_access_sha256 = (
        None if test_access is None else str(test_access["receipt_sha256"])
    )
    with args.experiment_config.open("r", encoding="utf-8") as stream:
        experiment_config = json.load(stream)
    try:
        residual_ready = validate_residual_ready(
            args.residual_ready,
            residual_manifest=args.residual_manifest,
            oof_ready=args.oof_ready,
            selected_partitions=selected_partitions,
        )
    except RuntimeError as error:
        raise SystemExit(str(error))
    residual_rows = load_jsonl(args.residual_manifest)
    try:
        validate_manifest_rows_against_frozen_learning_split(
            residual_rows,
            args.learning_split,
            require_episode_id=False,
            allowed_partitions=selected_partitions,
        )
    except LearningContractError as error:
        raise SystemExit(str(error))
    for row_number, source in enumerate(residual_rows, start=1):
        expected_receipt = (
            test_access_sha256 if source.get("partition") == "test" else None
        )
        if source.get("test_access_receipt_sha256") != expected_receipt:
            raise SystemExit(
                "residual row %d test-access receipt provenance mismatch" % row_number
            )
    buckets = residual_ready["validated_cohort"]
    manifest_case_ids = {str(row["case_id"]) for row in residual_rows}
    if len(manifest_case_ids) != len(residual_rows):
        raise SystemExit("residual manifest contains duplicate case_id")
    if manifest_case_ids != set(buckets["generated"]["case_ids"]):
        raise SystemExit("residual manifest differs from RESIDUAL_READY generated cohort")
    positive_from_rows = {
        str(row["case_id"])
        for row in residual_rows
        if int(row.get("fn_voxels", -1)) > 0
    }
    zero_from_rows = {
        str(row["case_id"])
        for row in residual_rows
        if int(row.get("fn_voxels", -1)) == 0
    }
    fp_positive_from_rows = {
        str(row["case_id"])
        for row in residual_rows
        if int(row.get("fp_voxels", -1)) > 0
    }
    zero_fp_from_rows = {
        str(row["case_id"])
        for row in residual_rows
        if int(row.get("fp_voxels", -1)) == 0
    }
    if positive_from_rows != set(buckets["fn_positive"]["case_ids"]):
        raise SystemExit("residual manifest FN-positive cohort differs from RESIDUAL_READY")
    if zero_from_rows != set(buckets["zero_fn"]["case_ids"]):
        raise SystemExit("residual manifest zero-FN cohort differs from RESIDUAL_READY")
    if fp_positive_from_rows != set(buckets["fp_positive"]["case_ids"]):
        raise SystemExit("residual manifest FP-positive cohort differs from RESIDUAL_READY")
    if zero_fp_from_rows != set(buckets["zero_fp"]["case_ids"]):
        raise SystemExit("residual manifest zero-FP cohort differs from RESIDUAL_READY")
    for row_number, source in enumerate(residual_rows, start=1):
        provenance = source.get("m0_provenance")
        if not isinstance(provenance, Mapping):
            raise SystemExit("residual row %d lacks M0 provenance" % row_number)
        if provenance.get("input_gt_sha256") != source.get("gt_sha256"):
            raise SystemExit(
                "residual row %d GT hash differs from OOF provenance" % row_number
            )
    generation = resolve_scribble_generation_contract(
        experiment_config,
        official_commit=args.official_commit,
        strategy_mode=args.strategy_mode,
        strategy_salt=args.strategy_salt,
        seed=args.seed,
    )
    crop_config = experiment_config["learning_tensor_normalization"]
    if any(os.path.lexists(str(path)) for path in (*final_directories, *final_files)):
        raise SystemExit("output paths must not already exist")
    simulator = load_official_simulator(
        args.official_simulator,
        expected_commit=generation["official_commit"],
        expected_sha256=generation["simulator_file_sha256"],
        runtime_manifest=args.official_runtime_manifest,
    )
    simulator_provenance = getattr(simulator, "_petct_official_provenance", None)
    validate_official_simulator_provenance(simulator_provenance, generation)
    # The trajectory corpus is M0 v6 OOF only: legacy OOF receipts fail closed.
    validated_oof = validate_m0_v6_oof_ready(args.oof_ready)
    rows: list[dict[str, Any]] = []
    trajectories: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    patient_partition: dict[str, str] = {}
    experiment_config_sha256 = sha256_file(args.experiment_config)
    local_radius_mm = float(experiment_config["editor"]["local_radius_mm"])
    minimum_local_area_mm2 = float(
        experiment_config["editor"]["minimum_local_area_mm2"]
    )
    visible_root, evaluation_root, authorized_root, state_root = final_directories
    exclusions_path, output_manifest, trajectories_path, ready_receipt = final_files
    requested_attempts: dict[str, dict[str, str]] = {}
    for source in residual_rows:
        case_id = str(source["case_id"])
        patient_id = str(source["patient_id"]).casefold()
        partition = str(source["partition"])
        for operation in ("ADD", "REMOVE"):
            for requested_strategy in selected_strategies(
                patient_id,
                generation["strategy_mode"],
                generation["strategy_salt"],
            ):
                attempt_id = scribble_attempt_id(
                    "natural", case_id, operation, requested_strategy
                )
                if attempt_id in requested_attempts:
                    raise RuntimeError("duplicate cue attempt id")
                requested_attempts[attempt_id] = {
                    "case_id": case_id,
                    "patient_id": patient_id,
                    "partition": partition,
                    "operation": operation,
                    "requested_strategy": requested_strategy,
                }

    def exclude_attempt(
        *,
        source: Mapping[str, Any],
        operation: str,
        requested_strategy: str,
        effective_strategy: str | None,
        reason: str,
        detail: str | None = None,
        round_index: int = 0,
        trajectory_id_: str | None = None,
    ) -> None:
        item = {
            "case_id": str(source["case_id"]),
            "patient_id": str(source["patient_id"]).casefold(),
            "partition": str(source["partition"]),
            "attempt_id": trajectory_attempt_id(
                str(source["case_id"]), operation, requested_strategy, round_index
            ),
            "operation": operation,
            "requested_strategy": requested_strategy,
            "effective_strategy": effective_strategy,
            "reason": reason,
            "round_index": int(round_index),
        }
        if detail:
            item["reason_detail"] = detail
        if trajectory_id_:
            item["trajectory_id"] = trajectory_id_
        exclusions.append(item)

    with staged_output_bundle(
        directory_outputs=final_directories, file_outputs=final_files
    ) as staged:
        staged_visible_root = staged[visible_root]
        staged_evaluation_root = staged[evaluation_root]
        staged_authorized_root = staged[authorized_root]
        staged_state_root = staged[state_root]
        for source in residual_rows:
            patient = str(source["patient_id"]).casefold()
            partition = str(source["partition"])
            if patient in patient_partition and patient_partition[patient] != partition:
                raise RuntimeError("patient crosses episode partitions")
            patient_partition[patient] = partition
            ct_path = _verified_source_path(source, "ct_path", "ct_sha256")
            pet_path = _verified_source_path(source, "pet_path", "pet_sha256")
            gt_path = _verified_source_path(source, "gt_path", "gt_sha256")
            m0_path = _verified_source_path(source, "m0_path", "m0_sha256")
            fn_path = _verified_source_path(source, "fn_path", "fn_sha256")
            fp_path = _verified_source_path(source, "fp_path", "fp_sha256")

            provenance = build_natural_oof_binding_from_validated(
                validated_oof,
                ready_path=args.oof_ready,
                case_id=source["case_id"],
                patient_id=patient,
                m0_path=m0_path,
                leaf_binding=source.get("truth_binding"),
            )
            if source.get("m0_provenance") != provenance:
                raise RuntimeError("residual manifest OOF provenance changed")

            gt_image = nib.load(str(gt_path))
            m0_image = nib.load(str(m0_path))
            fn_image = nib.load(str(fn_path))
            fp_image = nib.load(str(fp_path))
            images = (m0_image, fn_image, fp_image)
            if any(
                image.shape != gt_image.shape
                or not np.allclose(image.affine, gt_image.affine, atol=1e-3, rtol=0)
                for image in images
            ):
                raise RuntimeError("GT/M0/FN/FP geometry mismatch")
            gt_array = np.asarray(gt_image.dataobj)
            m0_array = np.asarray(m0_image.dataobj)
            residual_assets = {
                "ADD": {
                    "kind": "FN",
                    "path": fn_path,
                    "sha256": str(source["fn_sha256"]),
                    "mask": verify_full_residual(
                        gt_array, m0_array, np.asarray(fn_image.dataobj), operation="ADD"
                    ),
                },
                "REMOVE": {
                    "kind": "FP",
                    "path": fp_path,
                    "sha256": str(source["fp_sha256"]),
                    "mask": verify_full_residual(
                        gt_array,
                        m0_array,
                        np.asarray(fp_image.dataobj),
                        operation="REMOVE",
                    ),
                },
            }
            strategies = selected_strategies(
                patient,
                generation["strategy_mode"],
                generation["strategy_salt"],
            )
            for operation, asset in residual_assets.items():
                if not np.any(asset["mask"]):
                    for strategy in strategies:
                        exclude_attempt(
                            source=source,
                            operation=operation,
                            requested_strategy=strategy,
                            effective_strategy=None,
                            reason="EMPTY_%s_RESIDUAL" % asset["kind"],
                        )
                    continue
                for strategy in strategies:
                    _build_trajectory(
                        source=source,
                        patient=patient,
                        partition=partition,
                        ct_path=ct_path,
                        pet_path=pet_path,
                        gt_path=gt_path,
                        m0_path=m0_path,
                        fn_path=fn_path,
                        fp_path=fp_path,
                        gt_array=gt_array,
                        m0_array=m0_array,
                        gt_image=gt_image,
                        provenance=provenance,
                        operation=operation,
                        asset=asset,
                        strategy=strategy,
                        generation=generation,
                        simulator=simulator,
                        simulator_provenance=simulator_provenance,
                        local_radius_mm=local_radius_mm,
                        minimum_local_area_mm2=minimum_local_area_mm2,
                        crop_config=crop_config,
                        experiment_config_sha256=experiment_config_sha256,
                        test_access_sha256=test_access_sha256,
                        staged_visible_root=staged_visible_root,
                        staged_evaluation_root=staged_evaluation_root,
                        staged_authorized_root=staged_authorized_root,
                        staged_state_root=staged_state_root,
                        visible_root=visible_root,
                        evaluation_root=evaluation_root,
                        authorized_root=authorized_root,
                        state_root=state_root,
                        rows=rows,
                        trajectories=trajectories,
                        exclude_attempt=exclude_attempt,
                    )
        generated_attempt_ids = {
            str(row["attempt_id"]) for row in rows if int(row["round_index"]) == 0
        }
        round0_excluded = [
            row for row in exclusions if int(row["round_index"]) == 0
        ]
        excluded_attempt_ids = {str(row["attempt_id"]) for row in round0_excluded}
        all_excluded_ids = [str(row["attempt_id"]) for row in exclusions]
        if len(all_excluded_ids) != len(set(all_excluded_ids)):
            raise RuntimeError("one trajectory attempt received multiple exclusions")
        if len(excluded_attempt_ids) != len(round0_excluded):
            raise RuntimeError("one round-0 attempt received multiple exclusions")
        if generated_attempt_ids & excluded_attempt_ids:
            raise RuntimeError("round-0 attempt is both generated and excluded")
        if generated_attempt_ids | excluded_attempt_ids != set(requested_attempts):
            raise RuntimeError("round-0 attempt denominator is not closed")
        if not generated_attempt_ids:
            raise RuntimeError("no eligible trajectories were produced")
        generated_cases = {
            str(row["case_id"]) for row in trajectories
        }
        generated_patients = {
            str(row["patient_id"]).casefold() for row in trajectories
        }
        cases_with_excluded_attempts = {
            str(row["case_id"]) for row in exclusions
        }
        patients_with_excluded_attempts = {
            str(row["patient_id"]).casefold() for row in exclusions
        }
        selected_cases = {str(row["case_id"]) for row in residual_rows}
        selected_patients = {
            str(row["patient_id"]).casefold() for row in residual_rows
        }
        fully_excluded_cases = selected_cases - generated_cases
        fully_excluded_patients = {
            str(row["patient_id"]).casefold()
            for row in residual_rows
            if str(row["case_id"]) in fully_excluded_cases
        }
        partially_excluded_cases = cases_with_excluded_attempts & generated_cases
        partially_excluded_patients = {
            str(row["patient_id"]).casefold()
            for row in exclusions
            if str(row["case_id"]) in partially_excluded_cases
        }
        with staged[exclusions_path].open(
            "x", encoding="utf-8", newline="\n"
        ) as stream:
            for row in exclusions:
                stream.write(json.dumps(row, sort_keys=True) + "\n")
        with staged[output_manifest].open(
            "x", encoding="utf-8", newline="\n"
        ) as stream:
            for row in rows:
                stream.write(json.dumps(row, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        with staged[trajectories_path].open(
            "x", encoding="utf-8", newline="\n"
        ) as stream:
            for row in trajectories:
                stream.write(json.dumps(row, sort_keys=True) + "\n")
        round_count_histogram: dict[str, int] = {}
        terminations_by_reason: dict[str, int] = {}
        for trajectory in trajectories:
            key = str(trajectory["round_count"])
            round_count_histogram[key] = round_count_histogram.get(key, 0) + 1
            if trajectory.get("termination_reason"):
                reason = str(trajectory["termination_reason"])
                terminations_by_reason[reason] = (
                    terminations_by_reason.get(reason, 0) + 1
                )
        ready = {
            "schema_version": TRAJECTORY_READY_SCHEMA,
            "status": "PASS",
            "phase": TRAJECTORY_READY_PHASE,
            "lane": "natural",
            "strategy_mode": generation["strategy_mode"],
            "selected_partitions": sorted(selected_partitions),
            "inputs": {
                "residual_manifest": _output_file_record(
                    args.residual_manifest.resolve(),
                    args.residual_manifest.resolve(),
                ),
                "residual_ready": {
                    "path": residual_ready["ready_path"],
                    "bytes": Path(residual_ready["ready_path"]).stat().st_size,
                    "sha256": residual_ready["ready_sha256"],
                },
                "oof_ready": _output_file_record(
                    args.oof_ready.resolve(), args.oof_ready.resolve()
                ),
                "experiment_config": _output_file_record(
                    args.experiment_config.resolve(), args.experiment_config.resolve()
                ),
                "learning_split": _output_file_record(
                    args.learning_split.resolve(), args.learning_split.resolve()
                ),
                "official_source_provenance": dict(simulator_provenance),
            },
            "outputs": {
                "manifest": _output_file_record(staged[output_manifest], output_manifest),
                "trajectories": _output_file_record(
                    staged[trajectories_path], trajectories_path
                ),
                "exclusions": _output_file_record(
                    staged[exclusions_path], exclusions_path
                ),
                "visible": _tree_record(staged_visible_root, visible_root),
                "evaluation": _tree_record(staged_evaluation_root, evaluation_root),
                "authorized": _tree_record(staged_authorized_root, authorized_root),
                "states": _tree_record(staged_state_root, state_root),
            },
            "cohort": {
                "source": buckets["source"],
                "selected_source": _cohort_bucket(selected_cases, selected_patients),
                "eligible": _cohort_bucket(generated_cases, generated_patients),
                "excluded": _cohort_bucket(
                    fully_excluded_cases, fully_excluded_patients
                ),
                "partially_excluded": _cohort_bucket(
                    partially_excluded_cases, partially_excluded_patients
                ),
                "with_excluded_attempts": _cohort_bucket(
                    cases_with_excluded_attempts, patients_with_excluded_attempts
                ),
            },
            "attempts": {
                "requested_count": len(requested_attempts),
                "requested_ids": sorted(requested_attempts),
                "requested_ids_sha256": _canonical_hash(sorted(requested_attempts)),
                "generated_count": len(generated_attempt_ids),
                "generated_ids": sorted(generated_attempt_ids),
                "generated_ids_sha256": _canonical_hash(
                    sorted(generated_attempt_ids)
                ),
                "excluded_count": len(excluded_attempt_ids),
                "excluded_ids": sorted(excluded_attempt_ids),
                "excluded_ids_sha256": _canonical_hash(sorted(excluded_attempt_ids)),
            },
            "trajectory_stats": {
                "trajectories": len(trajectories),
                "episodes": len(rows),
                "round_count_histogram": round_count_histogram,
                "complete_5_rounds": sum(
                    1
                    for row in trajectories
                    if row["trajectory_status"] == STATUS_COMPLETE
                ),
                "residual_exhausted": sum(
                    1
                    for row in trajectories
                    if row["trajectory_status"] == STATUS_EXHAUSTED
                ),
                "truncated": sum(
                    1
                    for row in trajectories
                    if row["trajectory_status"] == STATUS_TRUNCATED
                ),
                "terminations_by_reason": terminations_by_reason,
                "round0_rows_sha256": _canonical_hash(
                    [
                        _single_round_projection(row)
                        for row in rows
                        if int(row["round_index"]) == 0
                    ]
                ),
                "round0_exclusions_sha256": _canonical_hash(
                    [
                        {
                            key: value
                            for key, value in row.items()
                            if key not in ("round_index", "trajectory_id")
                        }
                        for row in round0_excluded
                    ]
                ),
            },
            "parity_contract": {
                "single_round_corpus": PARITY_SINGLE_ROUND_DATASET,
                "only_difference": "round_count",
                "strategy_geometry": "identical_official_simulator_geometries",
                "sibling_structure": (
                    "at_most_three_strategy_siblings_per_case_m0_round_operation"
                ),
                "exclusion_rules": "identical_reason_codes",
                "lane_split": "identical_visible_label_audit",
                "round0_rows": "field-identical_to_single_round",
                "teacher_forcing": "ORACLE_AUTHORIZED_TARGET",
            },
            "exclusions_by_reason": _exclusion_summary(exclusions),
            "survivor_coverage": {
                "attempt_fraction": (
                    len(generated_attempt_ids) / len(requested_attempts)
                    if requested_attempts
                    else 0.0
                ),
                "case_fraction": (
                    len(generated_cases) / len(selected_cases)
                    if selected_cases
                    else 0.0
                ),
                "patient_fraction": (
                    len(generated_patients) / len(selected_patients)
                    if selected_patients
                    else 0.0
                ),
            },
            "experiment_result_count": 0,
            "thesis_citable": False,
            "locked_test_present": False,
        }
        ready["binding_sha256"] = _canonical_hash(ready)
        with staged[ready_receipt].open(
            "x", encoding="utf-8", newline="\n"
        ) as stream:
            stream.write(json.dumps(ready, ensure_ascii=False, indent=2, sort_keys=True))
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
    print(
        json.dumps(
            {
                "trajectories": len(trajectories),
                "episodes": len(rows),
                "excluded_attempts": len(exclusions),
                "strategies": generation["strategy_mode"],
                "ready_receipt": str(ready_receipt),
                "ready_receipt_sha256": sha256_file(ready_receipt),
            },
            sort_keys=True,
        )
    )
    return 0



if __name__ == "__main__":
    raise SystemExit(main())
