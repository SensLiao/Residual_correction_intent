#!/usr/bin/env python3
"""Build controlled P2T Pilot-6 states from two operation-specific scribbles.

The order is fixed: construct binary candidate masks, generate one official
foreground cue for the ADD triplet and one official background cue for the
REMOVE triplet, bind each cue relative to its three M0 states, and only then
render the six gold intents.  The two polarities are never merged into one
episode and no pseudo instance IDs are used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import nibabel as nib
import numpy as np


SCRIPTS = Path(__file__).resolve().parents[1]
for directory in (SCRIPTS, SCRIPTS / "common", SCRIPTS / "data", SCRIPTS / "p2t"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from common.petct_learning import load_jsonl, sha256_file  # noqa: E402
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
    apply_strategy_identity_policy,
    derive_goal_and_authorized_target,
    mask_fits_physical_crop,
    resolve_scribble_generation_contract,
    selected_strategies,
    staged_output_bundle,
    validate_official_simulator_provenance,
    write_binary_nifti,
)
from data.build_petct_scribble_episode import (  # noqa: E402
    DEFAULT_RUNTIME_MANIFEST,
    build_episode_documents,
    canonical_intent_frame,
    generate_residual_scribble,
    load_official_simulator,
    publish_episode_documents,
)
from data.materialize_petct_pilot6_states import (  # noqa: E402
    DATASET_ID,
    GOALS,
    GOALS_BY_OPERATION,
    construct_pilot6_states,
)
from data.validate_petct_learning_split import (  # noqa: E402
    load_and_validate_learning_split,
)


MATCHED_STATE_SCHEMA = "PETCT-P2T-MATCHED-STATE-v2.0"
CONTROLLED_READY_SCHEMA = "PETCT-CONTROLLED-P2T-DATA-READY-v2.0"
CONTROLLED_READY_PHASE = "CONTROLLED_MATCHED_STATE_PILOT6_MATERIALIZATION"
CONTROLLED_STAGE_ORDER = (
    "aligned_pet_ct_gt",
    "controlled_m0_candidate_states",
    "official_autopetv_operation_specific_scribble_on_shared_residual_support",
    "state_relative_lesion_binding",
    "canonical_intent_rendering",
)


def _mask_sha256(mask: np.ndarray) -> str:
    binary = np.asarray(mask, dtype=bool)
    header = json.dumps(
        {"axis_order": "xyz", "shape": list(binary.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(
        header + np.packbits(binary.reshape(-1)).tobytes()
    ).hexdigest()


def _group_id(case_id: str, strategy: str) -> str:
    digest = hashlib.sha256(
        f"PETCT-P2T-MATCHED-v2|{case_id}|{strategy}".encode("utf-8")
    ).hexdigest()
    return "matched-" + digest[:24]


def _episode_id(group_id: str, goal: str) -> str:
    digest = hashlib.sha256(
        f"PETCT-P2T-EPISODE-v2|{group_id}|{goal}".encode("utf-8")
    ).hexdigest()
    return "petct-" + digest[:24]


_VISIBLE_FORBIDDEN_FRAGMENTS = (
    "gt",
    "gold",
    "residual",
    "component",
    "authorized",
    "target",
    "source_case",
    "source_patient",
)


def _visible_safe_receipt(value: Any) -> Any:
    """Recursively strip evaluation-lane field names from a receipt subtree.

    Mirrors the visible-document firewall fragments so the controlled
    materializer receipt can be bound into the visible packet without
    leaking GT-derived field names (the full receipt stays in the
    evaluation document).
    """

    if isinstance(value, dict):
        output: dict[str, Any] = {}
        for key, child in value.items():
            if any(
                fragment in str(key).casefold()
                for fragment in _VISIBLE_FORBIDDEN_FRAGMENTS
            ):
                continue
            output[key] = _visible_safe_receipt(child)
        return output
    if isinstance(value, list):
        return [_visible_safe_receipt(child) for child in value]
    return value


def build_matched_state_six(
    pet: np.ndarray,
    ct: np.ndarray,
    gt: np.ndarray,
    *,
    spacing_xy: Sequence[float],
    strategy: str,
    simulator: Callable[..., Any],
    generation: Mapping[str, Any],
    local_radius_mm: float,
    minimum_local_area_mm2: float,
    learning_partition: str = "train",
    intent_renderer: Callable[[str], dict[str, Any]] = canonical_intent_frame,
) -> dict[str, Any]:
    """Create and independently re-derive all six counterfactual classes."""

    if learning_partition not in {"train", "val", "test"}:
        raise RuntimeError("learning_partition must be train, val, or test")
    # Euclidean scribble-anchored construction (D-2026-08-16-01): the v1
    # graph-distance partition disagreed systematically with the official
    # 15 mm / 50 mm^2 re-derivation (72.6% exclusion rate).  The official
    # scribbles are now generated FIRST on the ADD (FN) and REMOVE (FP)
    # masks, and every state mask is anchored to that Euclidean geometry, so
    # construction and re-derivation share one source of truth.
    from data.materialize_petct_pilot6_states_v2 import (
        PILOT6_V2_SCHEMA,
        Pilot6V2Error,
        construct_pilot6_states_v2,
        select_add_component,
        select_remove_component,
    )

    from data.materialize_petct_pilot6_states import THRESHOLDS

    try:
        add_component, _ = select_add_component(
            gt, min_component_voxels=int(THRESHOLDS["min_component_voxels"])
        )
        remove_component = select_remove_component(
            gt,
            add_component,
            shell_iterations=int(THRESHOLDS["remove_shell_iterations"]),
            min_component_voxels=int(THRESHOLDS["min_component_voxels"]),
        )
    except Pilot6V2Error as exc:
        return {
            "eligible": False,
            "reason": f"CONTROLLED_STATE_INELIGIBLE:{exc}",
            "receipt": {"schema_version": PILOT6_V2_SCHEMA, "status": "INELIGIBLE"},
        }

    scribbles = {
        operation: generate_residual_scribble(
            add_component if operation == "ADD" else remove_component,
            operation=operation,
            strategy=strategy,
            simulator=simulator,
            upstream_commit=str(generation["official_commit"]),
            seed=int(generation["seed"]),
        )
        for operation in ("ADD", "REMOVE")
    }
    try:
        v2 = construct_pilot6_states_v2(
            gt,
            add_component=add_component,
            remove_component=remove_component,
            scribble_add=scribbles["ADD"]["coordinates_xyz"],
            scribble_remove=scribbles["REMOVE"]["coordinates_xyz"],
            spacing_xy=spacing_xy,
            local_radius_mm=local_radius_mm,
            minimum_local_area_mm2=minimum_local_area_mm2,
        )
    except Pilot6V2Error as exc:
        return {
            "eligible": False,
            "reason": f"CONTROLLED_STATE_INELIGIBLE:{exc}",
            "receipt": {"schema_version": PILOT6_V2_SCHEMA, "status": "INELIGIBLE"},
        }
    states: dict[str, dict[str, Any]] = {}
    for expected_goal in GOALS:
        operation = expected_goal.split("_", 1)[0]
        scribble = scribbles[operation]
        state = v2["states"][expected_goal]
        # Same-source-of-truth insurance: with the Euclidean anchor the
        # re-derivation must always agree; a mismatch here is a system bug
        # and still fails closed.
        try:
            actual_goal, authorized, target_stats = derive_goal_and_authorized_target(
                gt=gt,
                m0=state["m0"],
                operation=operation,
                coordinates_xyz=scribble["coordinates_xyz"],
                spacing_xy=spacing_xy,
                local_radius_mm=local_radius_mm,
                minimum_local_area_mm2=minimum_local_area_mm2,
            )
        except RuntimeError as exc:
            return {
                "eligible": False,
                "reason": f"STATE_REDERIVATION_FAILED:{expected_goal}:{exc}",
                "receipt": {"schema_version": PILOT6_V2_SCHEMA, "status": "INELIGIBLE"},
                "scribbles": scribbles,
            }
        if actual_goal != expected_goal:
            return {
                "eligible": False,
                "reason": f"STATE_RELATIVE_GOAL_MISMATCH:{expected_goal}->{actual_goal}",
                "receipt": {"schema_version": PILOT6_V2_SCHEMA, "status": "INELIGIBLE"},
                "scribbles": scribbles,
            }
        states[expected_goal] = {
            "m0": np.asarray(state["m0"], dtype=np.uint8),
            "operation": operation,
            "operation_residual": np.asarray(
                state["operation_residual"], dtype=np.uint8
            ),
            "authorized_target": np.asarray(authorized, dtype=np.uint8),
            "target_stats": target_stats,
        }

    # Intent rendering is intentionally after the official scribble and binding.
    for goal in GOALS:
        states[goal]["gold_intent"] = intent_renderer(goal)
    return {
        "eligible": True,
        "scribbles": scribbles,
        "states": states,
        "receipt": {
            "schema_version": MATCHED_STATE_SCHEMA,
            "constructor_schema_version": PILOT6_V2_SCHEMA,
            "status": "ELIGIBLE",
            "learning_partition": learning_partition,
            "stage_order": list(CONTROLLED_STAGE_ORDER),
            "shared_physical_scribble_within_operation": True,
            "shared_physical_scribble_across_operations": False,
            "matched_goals": list(GOALS),
            "goals_by_operation": {
                operation: list(goals)
                for operation, goals in GOALS_BY_OPERATION.items()
            },
            "scribble_coordinate_sha256": {
                operation: scribble["coordinate_sha256"]
                for operation, scribble in scribbles.items()
            },
            "class_coverage_complete": set(states) == set(GOALS),
        },
    }


def _verified_image(row: Mapping[str, Any], key: str) -> tuple[Path, nib.Nifti1Image, str]:
    raw = Path(str(row.get(key) or ""))
    if raw.is_symlink():
        raise RuntimeError(f"missing regular non-symlink {key}: {raw}")
    path = raw.resolve()
    if not path.is_file():
        raise RuntimeError(f"missing regular {key}: {path}")
    digest = sha256_file(path)
    expected = row.get(key.replace("_path", "_sha256"))
    if expected is not None and expected != digest:
        raise RuntimeError(f"{key} hash differs from case manifest")
    return path, nib.load(str(path)), digest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-manifest", type=Path, required=True)
    parser.add_argument("--learning-split", type=Path, required=True)
    parser.add_argument("--experiment-config", type=Path, required=True)
    parser.add_argument("--official-simulator", type=Path, required=True)
    parser.add_argument(
        "--official-runtime-manifest",
        type=Path,
        default=DEFAULT_RUNTIME_MANIFEST,
    )
    parser.add_argument("--strategy-mode", choices=["primary", "all"])
    parser.add_argument(
        "--partitions",
        nargs="+",
        choices=["train", "val", "test"],
        required=True,
    )
    parser.add_argument("--state-root", type=Path, required=True)
    parser.add_argument("--visible-root", type=Path, required=True)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--exclusions", type=Path, required=True)
    parser.add_argument("--ready-receipt", type=Path, required=True)
    add_leaf_test_access_arguments(parser)
    args = parser.parse_args(argv)
    if len(args.partitions) != len(set(args.partitions)):
        parser.error("--partitions must not contain duplicates")
    final_dirs = [
        args.state_root.resolve(),
        args.visible_root.resolve(),
        args.evaluation_root.resolve(),
    ]
    final_files = [
        args.output_manifest.resolve(),
        args.exclusions.resolve(),
        args.ready_receipt.resolve(),
    ]
    selected_partitions = set(args.partitions)
    # Fail closed before the case manifest, frozen split, simulator, or any
    # PET/CT/GT volume is opened.
    try:
        test_access = enforce_partition_access(
            selected_partitions,
            receipt_path=args.test_access_receipt,
            experiment_config=args.experiment_config,
            learning_split=args.learning_split,
            run_root=args.run_root,
            output_paths=(*final_dirs, *final_files),
        )
    except TestAccessError as error:
        parser.error(str(error))
    with args.experiment_config.open("r", encoding="utf-8") as stream:
        experiment_config = json.load(stream)
    source_rows = load_jsonl(args.case_manifest)
    _, split_receipt = load_and_validate_learning_split(
        args.learning_split, source_rows, experiment_config
    )
    for row_number, source in enumerate(source_rows, start=1):
        case_id = str(source.get("case_id") or "")
        expected_partition = split_receipt["case_to_partition"].get(case_id)
        if expected_partition is None:
            raise RuntimeError(
                f"case manifest row {row_number} is absent from frozen learning split"
            )
        declared_partition = source.get("partition")
        if declared_partition is not None and declared_partition != expected_partition:
            raise RuntimeError(
                f"case manifest row {row_number} partition differs from frozen learning split"
            )
    test_access_sha256 = (
        None if test_access is None else str(test_access["receipt_sha256"])
    )
    generation = resolve_scribble_generation_contract(
        experiment_config, strategy_mode=args.strategy_mode
    )
    simulator = load_official_simulator(
        args.official_simulator,
        expected_commit=generation["official_commit"],
        expected_sha256=generation["simulator_file_sha256"],
        runtime_manifest=args.official_runtime_manifest,
    )
    provenance = getattr(simulator, "_petct_official_provenance", None)
    validate_official_simulator_provenance(provenance, generation)

    config_sha = sha256_file(args.experiment_config)
    local_radius = float(experiment_config["editor"]["local_radius_mm"])
    min_local_area = float(experiment_config["editor"]["minimum_local_area_mm2"])
    crop = experiment_config["learning_tensor_normalization"]
    rows: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    selected_sources = [
        source
        for source in source_rows
        if split_receipt["case_to_partition"][str(source["case_id"])]
        in selected_partitions
    ]
    requested_attempts: dict[str, dict[str, str]] = {}
    for source in selected_sources:
        case_id = str(source["case_id"])
        patient_id = str(source["patient_id"]).casefold()
        partition = split_receipt["case_to_partition"][case_id]
        for requested_strategy in selected_strategies(
            patient_id,
            generation["strategy_mode"],
            generation["strategy_salt"],
        ):
            attempt_id = _group_id(case_id, requested_strategy)
            if attempt_id in requested_attempts:
                raise RuntimeError("duplicate controlled scribble attempt id")
            requested_attempts[attempt_id] = {
                "case_id": case_id,
                "patient_id": patient_id,
                "partition": partition,
                "requested_strategy": requested_strategy,
            }

    def exclude_attempt(
        *,
        source: Mapping[str, Any],
        partition: str,
        requested_strategy: str,
        effective_strategy: str | None,
        reason: str,
        detail: str | None = None,
    ) -> None:
        item = {
            "case_id": str(source["case_id"]),
            "patient_id": str(source["patient_id"]).casefold(),
            "partition": partition,
            "attempt_id": _group_id(str(source["case_id"]), requested_strategy),
            "requested_strategy": requested_strategy,
            "effective_strategy": effective_strategy,
            "reason": reason,
        }
        if detail:
            item["reason_detail"] = detail
        exclusions.append(item)

    with staged_output_bundle(
        directory_outputs=final_dirs,
        file_outputs=final_files,
    ) as staged:
        state_stage, visible_stage, evaluation_stage = (
            staged[path] for path in final_dirs
        )
        for source in source_rows:
            case_id = str(source["case_id"])
            patient_id = str(source["patient_id"]).casefold()
            partition = split_receipt["case_to_partition"][case_id]
            if partition not in selected_partitions:
                continue
            try:
                ct_path, ct_image, ct_sha = _verified_image(source, "ct_path")
                pet_path, pet_image, pet_sha = _verified_image(source, "pet_path")
                gt_path, gt_image, gt_sha = _verified_image(source, "gt_path")
                images = (ct_image, pet_image, gt_image)
                if any(
                    image.shape != gt_image.shape
                    or not np.allclose(image.affine, gt_image.affine, atol=1e-3, rtol=0)
                    for image in images
                ):
                    raise RuntimeError("PET/CT/GT geometry mismatch")
                pet = np.asarray(pet_image.dataobj)
                ct = np.asarray(ct_image.dataobj)
                gt = (np.asarray(gt_image.dataobj) > 0).astype(np.uint8)
                held_out_fold = int(source["held_out_fold"])
                if held_out_fold not in range(5):
                    raise RuntimeError("held_out_fold must be 0..4")
                for strategy in selected_strategies(
                    patient_id, generation["strategy_mode"], generation["strategy_salt"]
                ):
                    attempt_id = _group_id(case_id, strategy)
                    try:
                        pilot6 = build_matched_state_six(
                            pet,
                            ct,
                            gt,
                            spacing_xy=gt_image.header.get_zooms()[:2],
                            strategy=strategy,
                            simulator=simulator,
                            generation=generation,
                            local_radius_mm=local_radius,
                            minimum_local_area_mm2=min_local_area,
                            learning_partition=partition,
                        )
                    except Exception as exc:
                        if generation["strategy_mode"] == "primary":
                            raise RuntimeError(
                                f"primary controlled attempt {attempt_id} failed closed: {exc}"
                            ) from exc
                        exclude_attempt(
                            source=source,
                            partition=partition,
                            requested_strategy=strategy,
                            effective_strategy=None,
                            reason="SCRIBBLE_GENERATION_FAILED",
                            detail=str(exc),
                        )
                        continue
                    if not pilot6["eligible"]:
                        exclude_attempt(
                            source=source,
                            partition=partition,
                            requested_strategy=strategy,
                            effective_strategy=None,
                            reason="CONTROLLED_STATE_INELIGIBLE",
                            detail=str(pilot6["reason"]),
                        )
                        continue
                    group_id = _group_id(case_id, strategy)
                    scribbles = pilot6["scribbles"]
                    dispositions = {
                        operation: apply_strategy_identity_policy(
                            scribble,
                            strategy_mode=generation["strategy_mode"],
                            context=(
                                f"primary controlled {operation} attempt {attempt_id}"
                            ),
                        )
                        for operation, scribble in scribbles.items()
                    }
                    crossed = next(
                        (value for value in dispositions.values() if value is not None),
                        None,
                    )
                    if crossed is not None:
                        exclude_attempt(
                            source=source,
                            partition=partition,
                            requested_strategy=strategy,
                            effective_strategy=crossed["effective_strategy"],
                            reason=crossed["reason"],
                            detail=crossed["detail"],
                        )
                        continue
                    if any(
                        not mask_fits_physical_crop(
                            pilot6["states"][goal]["authorized_target"][
                                :,
                                :,
                                scribbles[goal.split("_", 1)[0]]["source_slice"],
                            ],
                            center_xy=np.mean(
                                np.asarray(
                                    [
                                        [coordinate[0], coordinate[1]]
                                        for coordinate in scribbles[
                                            goal.split("_", 1)[0]
                                        ]["coordinates_xyz"]
                                    ]
                                ),
                                axis=0,
                            ),
                            spacing_xy=gt_image.header.get_zooms()[:2],
                            field_mm=float(crop["crop_field_mm"]),
                            output_size=int(crop["output_size_px"]),
                        )
                        for goal in GOALS
                    ):
                        exclude_attempt(
                            source=source,
                            partition=partition,
                            requested_strategy=strategy,
                            effective_strategy=str(
                                scribbles["ADD"]["effective_strategy"]
                            ),
                            reason="AUTHORIZED_TARGET_EXCEEDS_FROZEN_PHYSICAL_CROP",
                        )
                        continue
                    group_dir = state_stage / group_id
                    group_dir.mkdir()
                    final_group_dir = final_dirs[0] / group_id
                    for goal in GOALS:
                        episode_id = _episode_id(group_id, goal)
                        state = pilot6["states"][goal]
                        operation = str(state["operation"])
                        scribble = scribbles[operation]
                        goal_dir = group_dir / goal
                        goal_dir.mkdir()
                        final_goal_dir = final_group_dir / goal
                        m0_stage = goal_dir / "m0.nii.gz"
                        residual_stage = goal_dir / "operation_residual.nii.gz"
                        authorized_stage = goal_dir / "authorized.nii.gz"
                        write_binary_nifti(m0_stage, state["m0"], gt_image)
                        write_binary_nifti(
                            residual_stage, state["operation_residual"], gt_image
                        )
                        write_binary_nifti(authorized_stage, state["authorized_target"], gt_image)
                        m0_provenance = {
                            "kind": "controlled_matched_state",
                            "schema_version": MATCHED_STATE_SCHEMA,
                            "matched_state_group_id": group_id,
                            "goal": goal,
                            "operation": operation,
                            # The full pilot6 receipt carries GT-derived field names
                            # (gt_content_sha256, component thresholds, residual
                            # provenance) that the visible-lane firewall forbids.
                            # The complete receipt still reaches the evaluation
                            # document; the visible packet keeps only the
                            # opaque/eligible subset (2026-08-16 R5 fix).
                            "materializer_receipt": _visible_safe_receipt(
                                pilot6["receipt"]
                            ),
                        }
                        visible, evaluation = build_episode_documents(
                            episode_id=episode_id,
                            lane="controlled",
                            patient_group_hash=hashlib.sha256(
                                f"PETCT-PATIENT-GROUP-v2|{patient_id}".encode("utf-8")
                            ).hexdigest(),
                            montage_reference=f"learning-visible/{episode_id}.npz",
                            m0_provenance=m0_provenance,
                            scribble_record=scribble,
                            source_case_id=case_id,
                            source_patient_id=patient_id,
                            residual_sha256=_mask_sha256(
                                state["operation_residual"]
                            ),
                            residual_voxels=int(
                                state["operation_residual"].sum()
                            ),
                            gold_intent=state["gold_intent"],
                        )
                        doc_receipt = publish_episode_documents(
                            visible,
                            evaluation,
                            visible_root=visible_stage,
                            eval_root=evaluation_stage,
                        )
                        visible_final = final_dirs[1] / f"{episode_id}.json"
                        eval_final = final_dirs[2] / f"{episode_id}.json"
                        m0_final = final_goal_dir / "m0.nii.gz"
                        residual_final = final_goal_dir / "operation_residual.nii.gz"
                        authorized_final = final_goal_dir / "authorized.nii.gz"
                        rows.append(
                            {
                                "schema_version": MATCHED_STATE_SCHEMA,
                                "lane": "controlled_p2t",
                                "matched_state_group_id": group_id,
                                "operation": operation,
                                "target": state["target_stats"]["target"],
                                "scope": goal.rsplit("_", 1)[1],
                                "shared_physical_scribble_sha256": scribble[
                                    "coordinate_sha256"
                                ],
                                "shared_physical_scribble_scope": (
                                    "WITHIN_OPERATION_TRIPLET"
                                ),
                                "case_id": case_id,
                                "patient_id": patient_id,
                                "partition": partition,
                                "held_out_fold": held_out_fold,
                                "ct_path": str(ct_path),
                                "ct_sha256": ct_sha,
                                "pet_path": str(pet_path),
                                "pet_sha256": pet_sha,
                                "gt_path": str(gt_path),
                                "gt_sha256": gt_sha,
                                "m0_path": str(m0_final),
                                "m0_sha256": sha256_file(m0_stage),
                                "residual_kind": "FN" if operation == "ADD" else "FP",
                                "residual_path": str(residual_final),
                                "residual_sha256": sha256_file(residual_stage),
                                "residual_voxels": int(
                                    state["operation_residual"].sum()
                                ),
                                "residual_mask_sha256": _mask_sha256(
                                    state["operation_residual"]
                                ),
                                "authorized_path": str(authorized_final),
                                "authorized_sha256": sha256_file(authorized_stage),
                                "episode_id": episode_id,
                                "attempt_id": attempt_id,
                                "goal": goal,
                                "strategy": strategy,
                                "requested_strategy": scribble["requested_strategy"],
                                "effective_strategy": scribble["effective_strategy"],
                                "strategy_fallback": scribble["strategy_fallback"],
                                "scribble_density_mode": scribble[
                                    "scribble_density_mode"
                                ],
                                "strategy_mode": generation["strategy_mode"],
                                "strategy_salt": generation["strategy_salt"],
                                "strategy_assignment": generation["strategy_assignment"],
                                "seed": generation["seed"],
                                "coordinates_xyz": scribble["coordinates_xyz"],
                                "visible_document": str(visible_final),
                                "visible_document_sha256": doc_receipt["visible_sha256"],
                                "evaluation_document": str(eval_final),
                                "evaluation_document_sha256": doc_receipt["eval_sha256"],
                                "m0_provenance": m0_provenance,
                                "residual_contract": "CONTROLLED_MATCHED_STATE",
                                "official_source_provenance": dict(provenance),
                                "scribble_generation": {
                                    **generation,
                                    "stage_order": list(CONTROLLED_STAGE_ORDER),
                                    "attempt_id": attempt_id,
                                    "selected_strategy": strategy,
                                    "requested_strategy": scribble[
                                        "requested_strategy"
                                    ],
                                    "effective_strategy": scribble[
                                        "effective_strategy"
                                    ],
                                    "strategy_fallback": scribble[
                                        "strategy_fallback"
                                    ],
                                    "fallback_reason": scribble[
                                        "fallback_reason"
                                    ],
                                    "shared_across_goals": list(
                                        GOALS_BY_OPERATION[operation]
                                    ),
                                    "scribble_record": scribble,
                                },
                                "experiment_config_sha256": config_sha,
                                "learning_split_sha256": split_receipt["split_sha256"],
                                "test_access_receipt_sha256": (
                                    test_access_sha256 if partition == "test" else None
                                ),
                                "target_stats": state["target_stats"],
                            }
                        )
            except Exception as exc:
                raise RuntimeError(
                    f"controlled source integrity failed for {case_id}: {exc}"
                ) from exc
        group_counts: dict[str, int] = {}
        group_goals: dict[str, set[str]] = {}
        for row in rows:
            group_id = row["matched_state_group_id"]
            group_counts[group_id] = group_counts.get(group_id, 0) + 1
            group_goals.setdefault(group_id, set()).add(str(row["goal"]))
        if not group_counts:
            raise RuntimeError("no eligible controlled matched-state groups were produced")
        if any(count != len(GOALS) for count in group_counts.values()):
            raise RuntimeError("staged matched-state group is incomplete")
        if any(goals != set(GOALS) for goals in group_goals.values()):
            raise RuntimeError("staged matched-state group lacks exact six-class coverage")
        staged_group_ids = {path.name for path in state_stage.iterdir() if path.is_dir()}
        if staged_group_ids != set(group_counts):
            raise RuntimeError("staged state directories differ from complete groups")
        with staged[final_files[0]].open("x", encoding="utf-8", newline="\n") as stream:
            for row in rows:
                stream.write(json.dumps(row, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        with staged[final_files[1]].open("x", encoding="utf-8", newline="\n") as stream:
            for row in exclusions:
                stream.write(json.dumps(row, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

        generated_attempt_ids = {str(row["attempt_id"]) for row in rows}
        excluded_attempt_ids = {str(row["attempt_id"]) for row in exclusions}
        if len(excluded_attempt_ids) != len(exclusions):
            raise RuntimeError("one controlled attempt received multiple exclusions")
        if generated_attempt_ids & excluded_attempt_ids:
            raise RuntimeError("controlled attempt is both generated and excluded")
        if generated_attempt_ids | excluded_attempt_ids != set(requested_attempts):
            raise RuntimeError("controlled attempt denominator is not closed")
        if len(rows) != len(generated_attempt_ids) * len(GOALS):
            raise RuntimeError("controlled generated attempt is not a complete Pilot-6")
        selected_cases = {str(row["case_id"]) for row in selected_sources}
        selected_patients = {
            str(row["patient_id"]).casefold() for row in selected_sources
        }
        generated_cases = {str(row["case_id"]) for row in rows}
        generated_patients = {str(row["patient_id"]).casefold() for row in rows}
        cases_with_excluded_attempts = {
            str(row["case_id"]) for row in exclusions
        }
        patients_with_excluded_attempts = {
            str(row["patient_id"]).casefold() for row in exclusions
        }
        fully_excluded_cases = selected_cases - generated_cases
        fully_excluded_patients = {
            str(row["patient_id"]).casefold()
            for row in selected_sources
            if str(row["case_id"]) in fully_excluded_cases
        }
        partially_excluded_cases = cases_with_excluded_attempts & generated_cases
        partially_excluded_patients = {
            str(row["patient_id"]).casefold()
            for row in exclusions
            if str(row["case_id"]) in partially_excluded_cases
        }
        ready = {
            "schema_version": CONTROLLED_READY_SCHEMA,
            "status": "PASS",
            "phase": CONTROLLED_READY_PHASE,
            "lane": "controlled_p2t",
            "strategy_mode": generation["strategy_mode"],
            "selected_partitions": sorted(selected_partitions),
            "inputs": {
                "case_manifest": _output_file_record(
                    args.case_manifest.resolve(), args.case_manifest.resolve()
                ),
                "learning_split": _output_file_record(
                    args.learning_split.resolve(), args.learning_split.resolve()
                ),
                "experiment_config": _output_file_record(
                    args.experiment_config.resolve(), args.experiment_config.resolve()
                ),
                "official_source_provenance": dict(provenance),
            },
            "outputs": {
                "manifest": _output_file_record(
                    staged[final_files[0]], final_files[0]
                ),
                "exclusions": _output_file_record(
                    staged[final_files[1]], final_files[1]
                ),
                "states": _tree_record(state_stage, final_dirs[0]),
                "visible": _tree_record(visible_stage, final_dirs[1]),
                "evaluation": _tree_record(
                    evaluation_stage, final_dirs[2]
                ),
            },
            "cohort": {
                "source": _cohort_bucket(
                    {str(row["case_id"]) for row in source_rows},
                    {str(row["patient_id"]).casefold() for row in source_rows},
                ),
                "selected_source": _cohort_bucket(
                    selected_cases, selected_patients
                ),
                "eligible": _cohort_bucket(
                    generated_cases, generated_patients
                ),
                "excluded": _cohort_bucket(
                    fully_excluded_cases, fully_excluded_patients
                ),
                "partially_excluded": _cohort_bucket(
                    partially_excluded_cases, partially_excluded_patients
                ),
                "with_excluded_attempts": _cohort_bucket(
                    cases_with_excluded_attempts,
                    patients_with_excluded_attempts,
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
                "excluded_ids_sha256": _canonical_hash(
                    sorted(excluded_attempt_ids)
                ),
                "episodes": len(rows),
                "goals_per_generated_attempt": len(GOALS),
            },
            "intent_class_coverage": {
                "required_order": list(GOALS),
                "observed_counts": {
                    goal: sum(1 for row in rows if row["goal"] == goal)
                    for goal in GOALS
                },
                "complete": all(any(row["goal"] == goal for row in rows) for goal in GOALS),
                "instance_identity_source": (
                    "BINARY_GT_M0_18_CONNECTIVITY_NOT_INSTANCE_IDS"
                ),
            },
            "exclusions_by_reason": _exclusion_summary(exclusions),
            "survivor_coverage": {
                "attempt_fraction": len(generated_attempt_ids)
                / len(requested_attempts),
                "case_fraction": len(generated_cases) / len(selected_cases),
                "patient_fraction": len(generated_patients)
                / len(selected_patients),
            },
            "experiment_result_count": 0,
            "thesis_citable": False,
        }
        ready["binding_sha256"] = _canonical_hash(ready)
        with staged[final_files[2]].open(
            "x", encoding="utf-8", newline="\n"
        ) as stream:
            stream.write(json.dumps(ready, ensure_ascii=False, indent=2, sort_keys=True))
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())

    print(
        json.dumps(
            {
                "status": "MATERIALIZED",
                "lane": "controlled_p2t",
                "matched_state_groups": len(group_counts),
                "episodes": len(rows),
                "excluded": len(exclusions),
                "output_manifest_sha256": sha256_file(args.output_manifest),
                "ready_receipt": str(args.ready_receipt.resolve()),
                "ready_receipt_sha256": sha256_file(args.ready_receipt),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
