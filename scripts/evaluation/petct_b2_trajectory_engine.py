#!/usr/bin/env python3
"""Per-episode five-round trajectory engine for the B2 2D ceiling runner.

Holds the pure policy functions (arm conditioning table, residual-driven
gold call, VAL case guard) and the per-episode loop that advances a 3D state
volume through five rounds of plane-edit reconstruction.  The CLI/loading/
aggregation layer lives in ``run_petct_b2_trajectory_ceiling.py``; this
module never opens manifests itself so the two stay independently testable.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import nibabel as nib
import numpy as np
import torch

SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from common.petct_program_contract import (  # noqa: E402
    NEW_CUE_SENTINEL,
    family_to_id,
    protected_refs_policy,
    render_goal,
    validate_legal_call,
)
from common.petct_program_learning import (  # noqa: E402
    LearningContractError,
    _load_components,
    _load_visible_bundle,
    _sha256_file,
)
from common.petct_program_models import (  # noqa: E402
    NULL_FAMILY_ID,
    ProgramCompilerNet,
    ProgramEditorUNet2D,
)
from data.materialize_petct_learning_tensors import (  # noqa: E402
    physical_crop_resample_2d,
)
from evaluation.run_petct_v3_fullchain_official_test import (  # noqa: E402
    _restore_crop_slice_to_original,
    _runtime_components,
    _runtime_crop,
)
from evaluation.run_petct_w21_official_test import (  # noqa: E402
    _dice,
    choose_correction,
)

MATERIALIZED_AUTHORIZED = "MATERIALIZED_AUTHORIZED"
POLICY_ROUND_1 = "natural_label_oracle"
POLICY_ROUNDS_2_PLUS = "residual_driven_2d"
LEGAL_ARMS = ("J6", "J7", "J8", "J9")


def _require_arm_from_checkpoint(arm: str) -> str:
    if arm not in LEGAL_ARMS:
        raise LearningContractError(
            "editor checkpoint arm is not a legal J arm: %s" % arm
        )
    return arm


def _arm_conditioning(
    arm: str, family_id: int, operand_mode: int
) -> tuple[int, int, int, bool]:
    """Per-arm editor conditioning (family, operand, visual channels, embedding)."""
    _require_arm_from_checkpoint(arm)
    if arm == "J6":
        return NULL_FAMILY_ID, 2, 12, False
    if arm == "J7":
        return family_id, 0, 13, False
    if arm == "J8":
        return NULL_FAMILY_ID, 2, 13, True
    return family_id, operand_mode, 13, False


def _gold_call_residual_driven(
    operation: str,
    components: Sequence[Mapping[str, Any]],
    cue_hit_position: int | None,
) -> dict[str, Any]:
    """Derive the rounds-2+ ceiling call from the correction on the current state.

    REMOVE targets the deterministic cue-hit component (DELETE_COMPONENT).
    ADD extends the component with the largest scribble overlap on the
    prompted plane (COMPLETE_EXISTING; ties break by cue distance then
    position) or creates a new component when nothing overlaps (CREATE_NEW).
    """
    if operation not in ("ADD", "REMOVE"):
        raise LearningContractError("residual-driven call needs ADD or REMOVE")
    if operation == "REMOVE":
        if cue_hit_position is None or not 0 <= int(cue_hit_position) < len(components):
            raise LearningContractError(
                "residual-driven REMOVE lacks a current-state cue-hit component"
            )
        component = components[int(cue_hit_position)]
        family, operand = "DELETE_COMPONENT", str(component["component_key"])
        reason = "deterministic_cue_hit"
    else:
        hits = [
            (
                -float(component.get("cue_overlap_voxels") or 0.0),
                float(component.get("distance_from_cue_mm") or 0.0),
                int(component.get("candidate_position", 0)),
                component,
            )
            for component in components
            if float(component.get("cue_overlap_voxels") or 0.0) > 0.0
        ]
        if hits:
            component = min(hits, key=lambda value: value[:3])[3]
            family, operand = "COMPLETE_EXISTING", str(component["component_key"])
            reason = "max_cue_overlap_then_nearest_then_position"
        else:
            family, operand = "CREATE_NEW", NEW_CUE_SENTINEL
            reason = "no_component_overlap"
    validate_legal_call(operation, family, operand)
    return {
        "policy": POLICY_ROUNDS_2_PLUS,
        "operation": operation,
        "family": family,
        "operand": operand,
        "goal": render_goal(operation, family),
        "protected_refs": dict(protected_refs_policy(operation, operand)),
        "reason": reason,
    }


def _require_val_case(row: Mapping[str, Any]) -> str:
    """Refuse to join any case row that is not explicitly materialized VAL."""
    if str(row.get("partition") or "") == "test":
        raise LearningContractError("locked test case row must never be joined")
    if row.get("truth_materialization") != MATERIALIZED_AUTHORIZED:
        raise LearningContractError("case row is not materialized-authorized")
    return str(row["case_id"])


def _run_editor(
    *,
    model: ProgramEditorUNet2D,
    compiler: ProgramCompilerNet | None,
    arm: str,
    visual_17: np.ndarray,
    m0_slice: np.ndarray,
    selected_mask: np.ndarray,
    family: str,
    operand: str,
    operation_id: int,
    spacing_xy: np.ndarray,
    descriptor_vectors: np.ndarray | None,
    valid: np.ndarray | None,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """One editor forward under the frozen per-arm conditioning table."""
    family_id = family_to_id(family)
    operand_mode = 0 if operand != NEW_CUE_SENTINEL else 1
    family_for_editor, operand_for_editor, _, needs_embedding = _arm_conditioning(
        arm, family_id, operand_mode
    )
    signed_cue = visual_17[15:16] - visual_17[16:17]
    editor_visual = np.concatenate(
        [visual_17[:10], visual_17[12:13], signed_cue], axis=0
    )
    if arm != "J6":
        editor_visual = np.concatenate([editor_visual, selected_mask[None]], axis=0)
    state_embedding = None
    active_mask = None
    if needs_embedding:
        if compiler is None or descriptor_vectors is None or valid is None:
            raise LearningContractError("J8 requires the frozen compiler bundle")
        with torch.no_grad():
            outputs = compiler(
                torch.from_numpy(visual_17)[None].to(device),
                torch.from_numpy(spacing_xy)[None].to(device),
                torch.tensor([operation_id], device=device),
                torch.from_numpy(descriptor_vectors).to(device),
                torch.from_numpy(valid).to(device),
            )
        state_embedding = outputs["embedding"]
        active_mask = torch.tensor([True], device=device)
    with torch.no_grad():
        logits = model(
            torch.from_numpy(editor_visual)[None].to(device),
            torch.tensor([family_for_editor], device=device),
            torch.tensor([operand_for_editor], device=device),
            torch.tensor([0], device=device),
            state_embedding=state_embedding,
            active_mask=active_mask,
        )
    delta, corrected = ProgramEditorUNet2D.apply_program_operation(
        logits,
        torch.from_numpy(m0_slice)[None][None].float().to(device),
        torch.from_numpy(selected_mask)[None][None].float().to(device),
        torch.tensor([operation_id], device=device),
    )
    return (
        (delta[0, 0].cpu().numpy() > 0).astype(np.uint8),
        (corrected[0, 0].cpu().numpy() > 0).astype(np.uint8),
    )


def _selected_mask_from_call(
    *,
    call: Mapping[str, Any],
    keys: Sequence[str],
    masks: np.ndarray | None,
    cue_hit_position: int | None,
    shape: tuple[int, int],
) -> np.ndarray:
    operand = str(call["operand"])
    selected = np.zeros(shape, dtype=np.float32)
    if operand == NEW_CUE_SENTINEL:
        return selected
    try:
        position = keys.index(operand)
    except ValueError:
        raise LearningContractError("gold operand absent from candidates") from None
    if call["operation"] == "REMOVE" and position != cue_hit_position:
        raise LearningContractError("REMOVE gold operand is not the cue-hit component")
    if masks is None:
        raise LearningContractError("selected operand lacks crop-aligned mask")
    selected = np.asarray(masks[position], dtype=np.float32)
    if selected.shape != shape:
        raise LearningContractError("selected component geometry mismatch")
    return selected


def _plane_metrics(
    before_mask: np.ndarray, after_mask: np.ndarray, gt_plane: np.ndarray
) -> dict[str, float | None]:
    before = _dice(before_mask, gt_plane)
    after = _dice(after_mask, gt_plane)
    return {
        "dice_before": before,
        "dice_after": after,
        "delta_dice": None if before is None or after is None else after - before,
    }


def _patient_balanced_mean(
    per_round: dict[int, list[tuple[str, float]]],
) -> dict[str, float | None]:
    result: dict[str, float | None] = {}
    for round_index, pairs in per_round.items():
        patients: dict[str, list[float]] = {}
        for patient, value in pairs:
            patients.setdefault(patient, []).append(value)
        means = [float(np.mean(values)) for values in patients.values()]
        result[str(round_index)] = float(np.mean(means)) if means else None
    return result


def _mean_defined(values: Sequence[float | None]) -> float | None:
    present = [float(value) for value in values if value is not None]
    return float(np.mean(present)) if present else None


def run_episode(
    *,
    episode_id: str,
    row: Mapping[str, Any],
    rich: Mapping[str, Any],
    case_manifest: Mapping[str, dict[str, Any]],
    oracle_calls: Mapping[str, Mapping[str, Any]],
    candidates: Mapping[str, Mapping[str, Any]],
    simulator: Any,
    editor_model: ProgramEditorUNet2D,
    compiler: ProgramCompilerNet | None,
    arm: str,
    model_config: Mapping[str, Any],
    device: torch.device,
    case_cache: dict[str, Any],
    component_cache: dict[str, Any],
) -> dict[str, Any]:
    case_row = case_manifest.get(str(rich["case_id"]))
    if case_row is None:
        raise LearningContractError("episode case absent from the case manifest")
    _require_val_case(case_row)
    case_id = str(rich["case_id"])
    if case_id not in case_cache:
        for path_key, hash_key in (
            ("pet_path", "pet_sha256"),
            ("ct_path", "ct_sha256"),
        ):
            path = Path(str(case_row[path_key]))
            if _sha256_file(path) != str(case_row[hash_key]):
                raise LearningContractError("case volume hash mismatch: %s" % path)
        pet_image = nib.load(str(case_row["pet_path"]))
        ct_image = nib.load(str(case_row["ct_path"]))
        case_cache[case_id] = (
            np.asanyarray(pet_image.dataobj),
            np.asanyarray(ct_image.dataobj),
            pet_image.affine,
            ct_image.affine,
        )
    pet, ct, pet_affine, ct_affine = case_cache[case_id]
    source = rich["source_evaluation"]
    m0_path, gt_path = Path(str(source["m0_path"])), Path(str(source["gt_path"]))
    if _sha256_file(m0_path) != str(source["m0_sha256"]):
        raise LearningContractError("M0 hash mismatch: %s" % m0_path)
    if _sha256_file(gt_path) != str(source["gt_sha256"]):
        raise LearningContractError("GT hash mismatch: %s" % gt_path)
    m0_image = nib.load(str(m0_path))
    gt_image = nib.load(str(gt_path))
    shape = gt_image.shape
    for label, array, affine in (
        ("PET", pet, pet_affine),
        ("CT", ct, ct_affine),
        ("M0", np.asanyarray(m0_image.dataobj), m0_image.affine),
    ):
        if array.shape[:3] != shape or not np.allclose(affine, gt_image.affine):
            raise LearningContractError("%s %s/GT geometry differs" % (case_id, label))
    spacing_xy = np.asarray(gt_image.header.get_zooms()[:2], dtype=np.float32)
    spacing_xyz = np.asarray(
        [float(value) for value in gt_image.header.get_zooms()[:3]], dtype=np.float64
    )
    gt = (np.asanyarray(gt_image.dataobj) > 0).astype(np.uint8)
    state = (np.asanyarray(m0_image.dataobj) > 0).astype(np.uint8)
    geometry = rich["geometry"]
    center_z_round1 = int(rich["center_z"])
    operation_round1 = str(rich["operation"])
    oracle = oracle_calls.get(episode_id)
    if oracle is None:
        raise LearningContractError("missing oracle call for %s" % episode_id)
    if str(oracle.get("decision")) != "PREDICT":
        raise LearningContractError("oracle call for %s is not PREDICT" % episode_id)
    if str(oracle["operation"]) != operation_round1:
        raise LearningContractError("oracle call operation disagrees with the episode")
    validate_legal_call(
        str(oracle["operation"]), str(oracle["family"]), str(oracle["operand"])
    )

    bundle = _load_visible_bundle(dict(row), episode_id)
    evaluation_path = Path(str(rich["evaluation_npz"]))
    if _sha256_file(evaluation_path) != str(rich["evaluation_sha256"]):
        raise LearningContractError("evaluation bundle hash mismatch: %s" % episode_id)
    with np.load(str(evaluation_path), allow_pickle=False) as evaluation:
        if set(evaluation.files) != {"target", "authorized", "gt"}:
            raise LearningContractError("evaluation tensor schema mismatch")
        gt_crop_round1 = np.asarray(evaluation["gt"], dtype=np.float32) > 0

    foreground: set[tuple[int, int, int]] = set()
    background: set[tuple[int, int, int]] = set()
    rounds: list[dict[str, Any]] = []
    state_sha = hashlib.sha256(state.tobytes()).hexdigest()

    for round_index in range(1, int(model_config["trajectory_rounds"]) + 1):
        correction = None
        call = None
        operation_id = None
        if round_index == 1:
            call = {
                key: oracle[key]
                for key in (
                    "operation",
                    "family",
                    "operand",
                    "goal",
                    "oracle_selection",
                )
            }
            call["policy"] = POLICY_ROUND_1
            vectors, valid, masks, keys, cue_hit = _load_components(
                candidates, episode_id, component_cache
            )
            selected = _selected_mask_from_call(
                call=call,
                keys=keys,
                masks=masks,
                cue_hit_position=cue_hit,
                shape=tuple(bundle["m0"].shape),
            )
            visual_17 = np.asarray(bundle["visual"], dtype=np.float32)
            m0_slice = np.asarray(bundle["m0"], dtype=np.float32) > 0
            gt_crop = gt_crop_round1
            spacing_embed = np.asarray(bundle["spacing_xy"], dtype=np.float32)
            center_z = center_z_round1
            descriptors, valid_mask = vectors[None], valid[None]
        else:
            if np.any(gt):
                correction = choose_correction(
                    state, gt, strategy=str(rich["strategy"]), simulator=simulator
                )
                selected_set = {
                    tuple(int(value) for value in coordinate)
                    for coordinate in correction["coordinates"]
                }
                if correction["polarity"] == "foreground":
                    foreground.update(selected_set)
                else:
                    background.update(selected_set)
            if correction is not None and correction.get("coordinates"):
                coordinates = np.unique(
                    np.asarray(
                        [
                            [int(value) for value in c]
                            for c in correction["coordinates"]
                        ],
                        dtype=np.int64,
                    ),
                    axis=0,
                )
                z_values = np.unique(coordinates[:, 2])
                center_z = int(
                    z_values[
                        np.bincount(
                            np.searchsorted(z_values, coordinates[:, 2])
                        ).argmax()
                    ]
                )
                plane_coords = coordinates[coordinates[:, 2] == center_z]
                operation = (
                    "ADD" if correction["polarity"] == "foreground" else "REMOVE"
                )
                operation_id = 0 if operation == "ADD" else 1
                crop = _runtime_crop(
                    pet=pet,
                    ct=ct,
                    state=state,
                    coordinates=plane_coords,
                    center_z=center_z,
                    field_mm=float(model_config["field_mm"]),
                    output_size=int(model_config["output_size"]),
                    original_spacing_xy=spacing_xy,
                )
                crop["field_mm"] = float(model_config["field_mm"])
                crop["output_size"] = int(model_config["output_size"])
                visual_17 = np.asarray(crop["visual"], dtype=np.float32).copy()
                visual_17[15:16] = crop["scribble_slice"] if operation == "ADD" else 0
                visual_17[16:17] = (
                    crop["scribble_slice"] if operation == "REMOVE" else 0
                )
                components_bundle = _runtime_components(
                    state=state,
                    episode_key="b2-%s-state-%d" % (case_id, round_index),
                    m_sha256=state_sha,
                    spacing_xyz=spacing_xyz,
                    coordinates=plane_coords,
                    prompted_z=center_z,
                    crop=crop,
                    operation=operation,
                )
                call = _gold_call_residual_driven(
                    operation,
                    components_bundle["components"],
                    components_bundle["cue_hit_position"],
                )
                keys = [
                    str(component["component_key"])
                    for component in components_bundle["components"]
                ]
                selected = _selected_mask_from_call(
                    call=call,
                    keys=keys,
                    masks=(
                        np.stack(components_bundle["central_masks"], axis=0)
                        if components_bundle["central_masks"]
                        else None
                    ),
                    cue_hit_position=components_bundle["cue_hit_position"],
                    shape=tuple(crop["m0_slice"].shape),
                )
                m0_slice = np.asarray(crop["m0_slice"], dtype=np.float32) > 0
                gt_crop = (
                    physical_crop_resample_2d(
                        gt[:, :, center_z],
                        center_xy=crop["center_xy"],
                        spacing_xy=spacing_xy,
                        field_mm=float(model_config["field_mm"]),
                        output_size=int(model_config["output_size"]),
                        order=0,
                    )
                    > 0.5
                ).astype(np.float32)
                descriptors = components_bundle["descriptor_vectors"]
                valid_mask = components_bundle["valid"]
                spacing_embed = np.asarray(
                    [model_config["expected_spacing"]] * 2, dtype=np.float32
                )
            else:
                operation = None

        if call is None:
            rounds.append(
                {
                    "round": round_index,
                    "correction": correction,
                    "operation": None,
                    "gold_call": None,
                    "crop_plane": {
                        "dice_before": None,
                        "dice_after": None,
                        "delta_dice": None,
                    },
                    "state_plane": {
                        "dice_before": None,
                        "dice_after": None,
                        "delta_dice": None,
                    },
                    "state_sha256": state_sha,
                    "mixed_polarity": bool(foreground and background),
                }
            )
            continue
        family = str(call["family"])
        operand = str(call["operand"])
        operation = str(call["operation"])
        if operation_id is None:
            operation_id = 0 if operation == "ADD" else 1
        delta, corrected = _run_editor(
            model=editor_model,
            compiler=compiler,
            arm=arm,
            visual_17=visual_17,
            m0_slice=m0_slice,
            selected_mask=selected,
            family=family,
            operand=operand,
            operation_id=operation_id,
            spacing_xy=spacing_embed,
            descriptor_vectors=descriptors,
            valid=valid_mask,
            device=device,
        )
        if round_index == 1:
            restore_center_xy = np.asarray(
                geometry["crop_center_xy_voxel"], dtype=np.float32
            )
            restore_spacing = np.asarray(
                geometry["original_spacing_xy"], dtype=np.float32
            )
            restore_field = float(geometry["crop_field_mm"])
            restore_size = int(geometry["output_size_px"])
        else:
            restore_center_xy = np.asarray(crop["center_xy"], dtype=np.float32)
            restore_spacing = spacing_xy
            restore_field = float(model_config["field_mm"])
            restore_size = int(model_config["output_size"])
        state_plane_before = state[:, :, center_z].copy()
        restored = _restore_crop_slice_to_original(
            corrected,
            center_xy=restore_center_xy,
            original_spacing_xy=restore_spacing,
            field_mm=restore_field,
            output_size=restore_size,
            original_shape_xy=state.shape[:2],
        )
        state = state.copy()
        state[:, :, center_z] = restored
        state_sha = hashlib.sha256(state.tobytes()).hexdigest()
        gt_plane = gt[:, :, center_z]
        rounds.append(
            {
                "round": round_index,
                "correction": correction,
                "operation": operation,
                "gold_call": call,
                "scribble_voxels": (
                    len(str(rich["source_evaluation"]["scribble_coordinates_xyz"]))
                    if round_index == 1
                    else int(correction["selected_size"])
                ),
                "center_z": int(center_z),
                "crop_plane": _plane_metrics(m0_slice, corrected, gt_crop),
                "state_plane": _plane_metrics(
                    state_plane_before, state[:, :, center_z], gt_plane
                ),
                "state_sha256": state_sha,
                "mixed_polarity": bool(foreground and background),
            }
        )
    return {
        "episode_id": episode_id,
        "case_id": case_id,
        "patient_id": str(rich["patient_id"]),
        "strategy": str(rich["strategy"]),
        "rounds": rounds,
        "final_state_sha256": state_sha,
    }
