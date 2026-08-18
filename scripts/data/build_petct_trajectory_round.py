#!/usr/bin/env python3
"""Build one five-round teacher-forced trajectory (rounds 0..4).

Round 0 is the frozen single-round episode; rounds 1-4 advance the current
state by the oracle authorized target, draw the residual scribble with the
pinned simulator, and derive the gold goal with the identical state-relative
derivation.  Consumed only by ``build_petct_r13_trajectory_5r``.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Any, Mapping

import nibabel as nib
import numpy as np

SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))
for support_dir in (SCRIPTS_ROOT / "baseline", SCRIPTS_ROOT / "data"):
    if str(support_dir) not in sys.path:
        sys.path.insert(0, str(support_dir))

from common.petct_learning import sha256_file  # noqa: E402
from data.build_petct_scribble_dataset import (  # noqa: E402
    apply_strategy_identity_policy,
    classify_derivation_refusal,
    derive_goal_and_authorized_target,
    mask_fits_physical_crop,
    opaque_episode_id,
    write_binary_nifti,
)
from data.build_petct_scribble_episode import (  # noqa: E402
    CUE_INELIGIBLE_REASON,
    ResidualCueIneligibleError,
    _mask_sha256,
    build_episode_documents,
    canonical_intent_frame,
    generate_residual_scribble,
    publish_episode_documents,
)
from data.petct_trajectory_primitives import (  # noqa: E402
    MAX_ROUNDS,
    ROUND_RESIDUAL_CONTRACT,
    STATUS_COMPLETE,
    STATUS_EXHAUSTED,
    STATUS_TRUNCATED,
    TRAJECTORY_SUMMARY_SCHEMA,
    advance_trajectory_state,
    build_state_provenance,
    build_trajectory_round_documents,
    trajectory_attempt_id,
    trajectory_episode_id,
    trajectory_id,
)

def _build_trajectory(
    *,
    source: Mapping[str, Any],
    patient: str,
    partition: str,
    ct_path: Path,
    pet_path: Path,
    gt_path: Path,
    m0_path: Path,
    fn_path: Path,
    fp_path: Path,
    gt_array: np.ndarray,
    m0_array: np.ndarray,
    gt_image: nib.Nifti1Image,
    provenance: Mapping[str, Any],
    operation: str,
    asset: Mapping[str, Any],
    strategy: str,
    generation: Mapping[str, Any],
    simulator,
    simulator_provenance: Mapping[str, Any],
    local_radius_mm: float,
    minimum_local_area_mm2: float,
    crop_config: Mapping[str, Any],
    experiment_config_sha256: str,
    test_access_sha256: str | None,
    staged_visible_root: Path,
    staged_evaluation_root: Path,
    staged_authorized_root: Path,
    staged_state_root: Path,
    visible_root: Path,
    evaluation_root: Path,
    authorized_root: Path,
    state_root: Path,
    rows: list[dict[str, Any]],
    trajectories: list[dict[str, Any]],
    exclude_attempt,
) -> None:
    """Build one (case, operation, strategy) trajectory, rounds 0..4."""
    case_id = str(source["case_id"])
    tid = trajectory_id(case_id, operation, strategy)
    trajectory_rows: list[dict[str, Any]] = []
    episode_ids: list[str] = []
    state = (m0_array > 0).astype(np.uint8)
    state_sha256 = str(source["m0_sha256"])
    state_path: Path = m0_path
    base_m0_sha256 = state_sha256
    parent_state_sha256 = base_m0_sha256
    corrections: list[dict[str, Any]] = []
    status = STATUS_COMPLETE
    termination_reason: str | None = None
    residual = asset["mask"]
    residual_path: Path = asset["path"]
    residual_sha256 = str(asset["sha256"])
    round_count = 0
    while round_count < MAX_ROUNDS:
        round_index = round_count
        if round_index >= 1:
            state_staged = (
                staged_state_root / ("%s_round%d_state.nii.gz" % (tid, round_index))
            )
            write_binary_nifti(state_staged, state, gt_image)
            state_sha256 = sha256_file(state_staged)
            state_path = state_root / state_staged.name
            residual_staged = (
                staged_state_root
                / ("%s_round%d_residual.nii.gz" % (tid, round_index))
            )
            write_binary_nifti(residual_staged, residual, gt_image)
            residual_sha256 = sha256_file(residual_staged)
            residual_path = state_root / residual_staged.name
        try:
            record = generate_residual_scribble(
                residual,
                operation=operation,
                strategy=strategy,
                simulator=simulator,
                upstream_commit=generation["official_commit"],
                seed=generation["seed"],
                minimum_best_slice_pixels=generation["minimum_best_slice_pixels"],
            )
        except ResidualCueIneligibleError as exc:
            status = STATUS_TRUNCATED
            termination_reason = CUE_INELIGIBLE_REASON
            exclude_attempt(
                source=source,
                operation=operation,
                requested_strategy=strategy,
                effective_strategy=None,
                reason=CUE_INELIGIBLE_REASON,
                detail=str(exc),
                round_index=round_index,
                trajectory_id_=tid,
            )
            break
        except Exception as exc:
            if generation["strategy_mode"] == "primary":
                raise RuntimeError(
                    "primary trajectory cue attempt failed closed: %s" % exc
                ) from exc
            status = STATUS_TRUNCATED
            termination_reason = "CUE_GENERATION_FAILED"
            exclude_attempt(
                source=source,
                operation=operation,
                requested_strategy=strategy,
                effective_strategy=None,
                reason="CUE_GENERATION_FAILED",
                detail=str(exc),
                round_index=round_index,
                trajectory_id_=tid,
            )
            break
        disposition = apply_strategy_identity_policy(
            record,
            strategy_mode=generation["strategy_mode"],
            context="trajectory cue attempt %s round %d" % (tid, round_index),
        )
        if disposition is not None:
            status = STATUS_TRUNCATED
            termination_reason = disposition["reason"]
            exclude_attempt(
                source=source,
                operation=operation,
                requested_strategy=strategy,
                effective_strategy=disposition["effective_strategy"],
                reason=disposition["reason"],
                detail=disposition["detail"],
                round_index=round_index,
                trajectory_id_=tid,
            )
            break
        try:
            goal, authorized, target_stats = derive_goal_and_authorized_target(
                gt=gt_array,
                m0=state,
                operation=operation,
                coordinates_xyz=record["coordinates_xyz"],
                spacing_xy=gt_image.header.get_zooms()[:2],
                local_radius_mm=local_radius_mm,
                minimum_local_area_mm2=minimum_local_area_mm2,
            )
        except RuntimeError as exc:
            reason = classify_derivation_refusal(exc)
            status = STATUS_TRUNCATED
            termination_reason = reason
            exclude_attempt(
                source=source,
                operation=operation,
                requested_strategy=strategy,
                effective_strategy=str(record["effective_strategy"]),
                reason=reason,
                detail=str(exc),
                round_index=round_index,
                trajectory_id_=tid,
            )
            break
        center_xy = np.mean(
            np.asarray([[c[0], c[1]] for c in record["coordinates_xyz"]]), axis=0
        )
        center_z = int(record["coordinates_xyz"][0][2])
        if not mask_fits_physical_crop(
            authorized[:, :, center_z],
            center_xy=center_xy,
            spacing_xy=gt_image.header.get_zooms()[:2],
            field_mm=float(crop_config["crop_field_mm"]),
            output_size=int(crop_config["output_size_px"]),
        ):
            reason = "AUTHORIZED_TARGET_EXCEEDS_FROZEN_PHYSICAL_CROP"
            status = STATUS_TRUNCATED
            termination_reason = reason
            exclude_attempt(
                source=source,
                operation=operation,
                requested_strategy=strategy,
                effective_strategy=str(record["effective_strategy"]),
                reason=reason,
                round_index=round_index,
                trajectory_id_=tid,
            )
            break
        attempt_id = trajectory_attempt_id(case_id, operation, strategy, round_index)
        state_derivation: dict[str, Any] = {}
        round_provenance = dict(provenance)
        if round_index >= 1:
            round_provenance = build_state_provenance(
                trajectory_id=tid,
                round_index=round_index,
                operation=operation,
                state_path=state_path,
                state_sha256=state_sha256,
                base_m0_sha256=base_m0_sha256,
                parent_state_sha256=parent_state_sha256,
                corrections=corrections,
                input_ct_sha256=str(provenance.get("input_ct_sha256") or ""),
                input_pet_sha256=str(provenance.get("input_pet_sha256") or ""),
                held_out_fold=int(source["held_out_fold"]),
            )
        generation_receipt = {
            **generation,
            "attempt_id": attempt_id,
            "operation": operation,
            "residual_kind": asset["kind"],
            "selected_strategy": strategy,
            "requested_strategy": record["requested_strategy"],
            "effective_strategy": record["effective_strategy"],
            "strategy_fallback": record["strategy_fallback"],
            "fallback_reason": record["fallback_reason"],
            "strategy_audit": record["strategy_audit"],
            "official_source_provenance": dict(simulator_provenance),
            "m0_provenance": round_provenance,
            "cue_contract_version": record["contract_version"],
            "simulator_entrypoint": record["simulator_entrypoint"],
            "residual_artifact_path": str(residual_path),
            "residual_artifact_sha256": residual_sha256,
            "residual_mask_sha256": record["residual_sha256"],
            "residual_voxels": record["residual_voxels"],
            "cue_eligibility": record["cue_eligibility"],
            "eligible_residual_sha256": record["eligible_residual_sha256"],
            "coordinate_count": record["coordinate_count"],
            "coordinate_sha256": record["coordinate_sha256"],
            "source_slice": record["source_slice"],
            "source_component_area": record["source_component_area"],
            "cue_polarity": record["polarity"],
            "single_component_connectivity": 18,
            "local_radius_mm": local_radius_mm,
            "minimum_local_area_mm2": minimum_local_area_mm2,
            "crop_field_mm": float(crop_config["crop_field_mm"]),
            "crop_output_size_px": int(crop_config["output_size_px"]),
            "crop_output_spacing_mm": float(crop_config["output_spacing_mm"]),
            "experiment_config_sha256": experiment_config_sha256,
        }
        if round_index >= 1:
            generation_receipt.update(
                {
                    "trajectory_id": tid,
                    "round_index": round_index,
                    "state_sha256": state_sha256,
                    "teacher_forcing": "ORACLE_AUTHORIZED_TARGET",
                }
            )
        record["official_source_provenance"] = dict(simulator_provenance)
        record["generation_receipt"] = generation_receipt
        episode_id = (
            opaque_episode_id(case_id, goal, strategy)
            if round_index == 0
            else trajectory_episode_id(tid, round_index)
        )
        # Round 0 keeps the frozen single-round authorized file naming so the
        # parity contract holds field-for-field on the row's authorized_path.
        authorized_staged = (
            staged_authorized_root / ("%s_authorized.nii.gz" % episode_id)
            if round_index == 0
            else staged_authorized_root
            / ("%s_round%d_authorized.nii.gz" % (tid, round_index))
        )
        write_binary_nifti(authorized_staged, authorized, gt_image)
        authorized_sha256 = sha256_file(authorized_staged)
        authorized_final_path = authorized_root / authorized_staged.name
        generation_receipt.update(
            {
                "goal": goal,
                "target_stats": target_stats,
                "authorized_target_sha256": authorized_sha256,
            }
        )
        patient_hash = hashlib.sha256(
            ("PETCT-PATIENT-GROUP-v2|" + patient).encode("utf-8")
        ).hexdigest()
        if round_index == 0:
            visible, evaluation = build_episode_documents(
                episode_id=episode_id,
                lane="natural",
                patient_group_hash=patient_hash,
                montage_reference="learning-visible/%s.npz" % episode_id,
                m0_provenance=provenance,
                scribble_record=record,
                source_case_id=case_id,
                source_patient_id=patient,
                residual_sha256=record["residual_sha256"],
                residual_voxels=record["residual_voxels"],
                gold_intent=canonical_intent_frame(goal),
            )
        else:
            state_derivation = {
                "contract": "DERIVED_FROM_TEACHER_FORCED_STATE_AND_GENERATED_SCRIBBLE",
                "trajectory_id": tid,
                "round_index": round_index,
                "goal": goal,
                "authorized_target_sha256": _mask_sha256(authorized),
                "authorized_target_voxels": int(np.asarray(authorized).sum()),
                "spacing_xy_mm": [
                    float(value) for value in gt_image.header.get_zooms()[:2]
                ],
                "local_radius_mm": local_radius_mm,
                "minimum_local_area_mm2": minimum_local_area_mm2,
                "experiment_config_sha256": experiment_config_sha256,
                "target_stats": target_stats,
            }
            visible, evaluation = build_trajectory_round_documents(
                episode_id=episode_id,
                trajectory_id=tid,
                round_index=round_index,
                lane="natural",
                patient_group_hash=patient_hash,
                montage_reference="learning-visible/%s.npz" % episode_id,
                state_provenance=round_provenance,
                scribble_record=record,
                source_case_id=case_id,
                source_patient_id=patient,
                residual_sha256=residual_sha256,
                residual_voxels=int(record["residual_voxels"]),
                gold_intent=canonical_intent_frame(goal),
                state_relative_derivation=state_derivation,
            )
        receipt = publish_episode_documents(
            visible,
            evaluation,
            visible_root=staged_visible_root,
            eval_root=staged_evaluation_root,
        )
        visible_final_path = visible_root / ("%s.json" % episode_id)
        evaluation_final_path = evaluation_root / ("%s.json" % episode_id)
        row = {
            **{key: source[key] for key in ("case_id", "patient_id", "partition", "held_out_fold")},
            "round_index": round_index,
            "ct_path": str(ct_path),
            "pet_path": str(pet_path),
            "m0_path": str(state_path),
            "gt_path": str(gt_path),
            **{
                key: source[key]
                for key in (
                    "ct_sha256",
                    "pet_sha256",
                    "gt_sha256",
                    "learning_split_sha256",
                )
            },
            "m0_sha256": state_sha256,
            "authorized_path": str(authorized_final_path),
            "authorized_sha256": authorized_sha256,
            "episode_id": episode_id,
            "attempt_id": attempt_id,
            "goal": goal,
            "operation": operation,
            "target": target_stats["target"],
            "scope": goal.rsplit("_", 1)[1],
            "cue_polarity": record["polarity"],
            "strategy": strategy,
            "requested_strategy": record["requested_strategy"],
            "effective_strategy": record["effective_strategy"],
            "strategy_fallback": record["strategy_fallback"],
            "strategy_mode": generation["strategy_mode"],
            "strategy_salt": generation["strategy_salt"],
            "strategy_assignment": generation["strategy_assignment"],
            "seed": generation["seed"],
            "coordinates_xyz": record["coordinates_xyz"],
            "visible_document": str(visible_final_path),
            "visible_document_sha256": receipt["visible_sha256"],
            "evaluation_document": str(evaluation_final_path),
            "evaluation_document_sha256": receipt["eval_sha256"],
            "scribble_density_mode": record["scribble_density_mode"],
            "fallback_mode": record["fallback_mode"],
            "m0_provenance": round_provenance,
            "residual_kind": asset["kind"],
            "residual_path": str(residual_path),
            "residual_sha256": residual_sha256,
            "residual_voxels": int(record["residual_voxels"]),
            "residual_mask_sha256": str(record["residual_sha256"]),
            "fn_path": str(fn_path),
            "fn_sha256": str(source["fn_sha256"]),
            "fp_path": str(fp_path),
            "fp_sha256": str(source["fp_sha256"]),
            "residual_contract": (
                source.get("residual_contract")
                if round_index == 0
                else ROUND_RESIDUAL_CONTRACT
            ),
            "official_source_provenance": dict(simulator_provenance),
            "scribble_generation": generation_receipt,
            "experiment_config_sha256": experiment_config_sha256,
            "test_access_receipt_sha256": (
                test_access_sha256 if partition == "test" else None
            ),
            "target_stats": target_stats,
        }
        trajectory_rows.append(row)
        episode_ids.append(episode_id)
        round_count += 1
        corrections.append(
            {
                "round_index": round_index,
                "operation": operation,
                "authorized_sha256": authorized_sha256,
            }
        )
        parent_state_sha256 = state_sha256
        state, residual, exhausted = advance_trajectory_state(
            gt_array, state, authorized, operation=operation
        )
        if exhausted:
            status = STATUS_EXHAUSTED
            termination_reason = (
                "EMPTY_FN_RESIDUAL" if operation == "ADD" else "EMPTY_FP_RESIDUAL"
            )
            break
    for row in trajectory_rows:
        row["trajectory_id"] = tid
        row["round_count"] = round_count
        row["trajectory_status"] = status
        row["termination_reason"] = termination_reason
    rows.extend(trajectory_rows)
    trajectories.append(
        {
            "schema_version": TRAJECTORY_SUMMARY_SCHEMA,
            "trajectory_id": tid,
            "case_id": case_id,
            "patient_id": patient,
            "partition": partition,
            "operation": operation,
            "strategy": strategy,
            "round_count": round_count,
            "trajectory_status": status,
            "termination_reason": termination_reason,
            "episode_ids": episode_ids,
            "round0_episode_id": episode_ids[0],
            "base_m0_sha256": base_m0_sha256,
            "final_state_sha256": state_sha256,
            "final_residual_voxels": int(np.asarray(residual).sum()),
        }
    )

