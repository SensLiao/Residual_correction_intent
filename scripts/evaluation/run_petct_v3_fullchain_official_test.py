#!/usr/bin/env python3
"""v3 full-chain official TEST runner (compiler -> editor, five corrections).

Formal test execution is possible only through a consumed exactly-once
test-access receipt (common/petct_test_access.py).  For each locked-test case
the runner reproduces the official autoPET V six-state protocol: state 0 is
the M0 current state and states 1..5 are the volumes after five accumulated
corrections.  Every correction runs the real chain — runtime component
enumeration, predicted legal program (compiler), program-conditioned editor,
SCEP algebra — and advances a full 3D state volume.

The v3 editor is a 2D prompted-plane editor, so the runner records three
labelled metric domains and never mixes them:
  * ``3d_full_volume``        official-protocol domain (plane-edit
                              reconstruction); headline-citable;
  * ``2d_prompted_plane``     supplementary; never merged with 3D;
  * ``single_slice_ceiling``  reference ceiling (GT plane substituted);
                              never a model result.

Volume/cue/checkpoint-binding policies are recorded per case and flagged
``pending_formal_amendment`` until the director freezes the formal closed-loop
protocol (tasks-dashboard "Needs Human Decision" 1/2).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import nibabel as nib
import numpy as np
import torch

SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from common.petct_components import (  # noqa: E402
    enumerate_components,
    cue_hit_component_position,
)
from common.petct_mainline_lineage import (  # noqa: E402
    MAINLINE_SOURCE,
    validate_r13_lineage_receipt,
)
from common.petct_program_contract import (  # noqa: E402
    NEW_CUE_SENTINEL,
    GRAMMAR_VERSION,
    family_to_id,
    render_goal,
    protected_refs_policy,
)
from common.petct_program_models import (  # noqa: E402
    COMPONENT_DESCRIPTOR_DIM,
    LegalCallCompiler,
    ProgramCompilerNet,
    ProgramEditorUNet2D,
)
from common.petct_test_access import (  # noqa: E402
    TestAccessError,
    add_leaf_test_access_arguments,
    enforce_partition_access,
)
from common.petct_w21_test_access import (  # noqa: E402
    PROTOCOL,
)
from data.materialize_petct_learning_tensors import (  # noqa: E402
    axial_stack,
    normalize_ct,
    normalize_pet,
    physical_crop_resample_2d,
)
from evaluation.run_petct_w21_official_test import (  # noqa: E402
    _dice,
    _file_record,
    _load_module,
    _mean,
    _regular,
    _seal,
    _validate_pinned_official_smoke_code,
    _verify_seal,
    _write_json_exclusive,
    choose_correction,
    compute_state_metrics,
)

CASE_SCHEMA = "PETCT-V3-FULLCHAIN-OFFICIAL-TEST-CASE-v1.0"
ARM_SCHEMA = "PETCT-V3-FULLCHAIN-OFFICIAL-TEST-ARM-v1.0"
COMPILER_CHECKPOINT_SCHEMA = "PETCT-PROGRAM-COMPILER-CHECKPOINT-v1.0"
EDITOR_CHECKPOINT_SCHEMA = "PETCT-PROGRAM-EDITOR-CHECKPOINT-v1.0"
RUNNER_SOURCE_SHA256 = "runner_source_sha256"
PLANE_POLICY = "scribble_mode_z"
VOLUME_POLICY = "plane_edit_reconstruction"
CUE_POLICY = "official_accumulation_projected_to_prompted_plane"
CHECKPOINT_BINDING_POLICY = (
    "schema_architecture_lineage_bound_training_manifest_recorded_not_required"
)

PENDING_AMENDMENT_FIELDS = {
    "state_semantics": "state_0=M0; state_i=plane-edit reconstruction after i corrections",
    "plane_policy": PLANE_POLICY,
    "volume_policy": VOLUME_POLICY,
    "cue_policy": CUE_POLICY,
    "checkpoint_binding_policy": CHECKPOINT_BINDING_POLICY,
    "pending_formal_amendment": True,
    "note": (
        "Formal closed-loop / estimator-matched protocol amendment is a "
        "pending director decision (tasks-dashboard Needs Human Decision 1/2). "
        "This runner is protocol-ready code, not a scientific result."
    ),
}


class V3FullChainError(RuntimeError):
    """Raised when the full-chain run violates a frozen contract."""


def _verified_case_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one case-manifest row and its hash-bound input files."""

    for key in ("case_id", "patient_id", "pet_path", "ct_path", "gt_path", "m0_path"):
        if not row.get(key):
            raise V3FullChainError(f"case row missing {key}")
    for key in ("pet_sha256", "ct_sha256", "gt_sha256", "m0_sha256"):
        if not row.get(key):
            raise V3FullChainError(f"case {row['case_id']} missing {key}")
    for path_key, hash_key in (
        ("pet_path", "pet_sha256"),
        ("ct_path", "ct_sha256"),
        ("gt_path", "gt_sha256"),
        ("m0_path", "m0_sha256"),
    ):
        record = _file_record(Path(str(row[path_key])), label=path_key)
        if record["sha256"] != str(row[hash_key]):
            raise V3FullChainError(
                f"case {row['case_id']} {path_key} sha256 differs from manifest"
            )
    return dict(row)


def _load_compiler_bundle(
    checkpoint_path: Path, lineage_receipt_path: Path, device: torch.device
) -> tuple[ProgramCompilerNet, LegalCallCompiler, dict[str, Any]]:
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    if checkpoint.get("schema_version") != COMPILER_CHECKPOINT_SCHEMA:
        raise V3FullChainError("compiler checkpoint schema is not the frozen v1.0")
    if checkpoint.get("architecture_id") != "matched_legal_component_program_v1":
        raise V3FullChainError("compiler checkpoint architecture id differs")
    lineage = validate_r13_lineage_receipt(lineage_receipt_path)
    if checkpoint.get("source_m0_lineage") != MAINLINE_SOURCE:
        raise V3FullChainError("compiler checkpoint source M0 lineage differs")
    if checkpoint.get("lineage_receipt_sha256") != lineage["receipt_sha256"]:
        raise V3FullChainError("compiler checkpoint lineage receipt differs")
    include_repair = bool(checkpoint["hyperparameters"]["include_repair"])
    model = ProgramCompilerNet(include_repair=include_repair).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    compiler = LegalCallCompiler(include_repair=include_repair)
    return model, compiler, checkpoint


def _load_editor_bundle(
    checkpoint_path: Path, device: torch.device
) -> tuple[ProgramEditorUNet2D, dict[str, Any]]:
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    if checkpoint.get("schema_version") != EDITOR_CHECKPOINT_SCHEMA:
        raise V3FullChainError("editor checkpoint schema is not the frozen v1.0")
    if checkpoint.get("architecture_id") != "matched_legal_component_program_editor_v1":
        raise V3FullChainError("editor checkpoint architecture id differs")
    if checkpoint.get("arm") not in ("J9", "J9C", "J8", "J6", "J7"):
        raise V3FullChainError(f"editor checkpoint arm is not a legal J arm: {checkpoint.get('arm')}")
    if checkpoint.get("call_source") != "gold":
        raise V3FullChainError("full-chain editor checkpoint must be gold-call trained")
    model = ProgramEditorUNet2D(conditioner="program").to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    return model, checkpoint


def _runtime_crop(
    *,
    pet: np.ndarray,
    ct: np.ndarray,
    state: np.ndarray,
    coordinates: np.ndarray,
    center_z: int,
    field_mm: float,
    output_size: int,
    original_spacing_xy: np.ndarray,
) -> dict[str, Any]:
    """Build the compiler 17-channel visible crop exactly as the training-time
    materializer does, with the CURRENT state volume in place of M0."""

    # Frozen training-time normalization contracts (docs/05 section 4.1).
    ct = normalize_ct(np.asarray(ct, dtype=np.float32))  # clip[-1000,1000]/1000
    pet = normalize_pet(np.asarray(pet, dtype=np.float32))[0]  # log1p+median/IQR
    scribble = np.zeros(state.shape, dtype=np.uint8)
    for coordinate in coordinates:
        scribble[tuple(int(value) for value in coordinate)] = 1
    center_xy = np.mean(
        np.asarray([[float(coord[0]), float(coord[1])] for coord in coordinates]),
        axis=0,
    )
    raw_visual = np.concatenate(
        [
            axial_stack(pet, center_z),
            axial_stack(ct, center_z),
            axial_stack(state, center_z),
            scribble[:, :, center_z][None],
            np.zeros((1, *scribble.shape[:2]), dtype=np.float32),
        ],
        axis=0,
    ).astype(np.float32)
    visual = np.concatenate(
        [
            physical_crop_resample_2d(
                raw_visual[:10],
                center_xy=center_xy,
                spacing_xy=original_spacing_xy,
                field_mm=field_mm,
                output_size=output_size,
                order=1,
            ),
            physical_crop_resample_2d(
                raw_visual[10:],
                center_xy=center_xy,
                spacing_xy=original_spacing_xy,
                field_mm=field_mm,
                output_size=output_size,
                order=0,
            ),
        ],
        axis=0,
    ).astype(np.float32)
    m0_slice = physical_crop_resample_2d(
        state[:, :, center_z],
        center_xy=center_xy,
        spacing_xy=original_spacing_xy,
        field_mm=field_mm,
        output_size=output_size,
        order=0,
    ).astype(np.uint8)
    scribble_slice = physical_crop_resample_2d(
        scribble[:, :, center_z],
        center_xy=center_xy,
        spacing_xy=original_spacing_xy,
        field_mm=field_mm,
        output_size=output_size,
        order=0,
    ).astype(np.uint8)
    if not scribble_slice.any():
        raise V3FullChainError("resampling lost the official scribble")
    return {
        "visual": visual,
        "m0_slice": m0_slice,
        "scribble_slice": scribble_slice,
        "center_xy": center_xy,
        "spacing_xy": np.asarray(original_spacing_xy, dtype=np.float32),
    }


def _restore_crop_slice_to_original(
    crop_slice: np.ndarray,
    *,
    center_xy: np.ndarray,
    original_spacing_xy: np.ndarray,
    field_mm: float,
    output_size: int,
    original_shape_xy: Sequence[int],
) -> np.ndarray:
    """Exact inverse of ``physical_crop_resample_2d`` (nearest, zero-fill).

    Plane-edit reconstruction policy: the corrected crop-space slice is
    resampled back onto the original X/Y grid before replacing the prompted
    axial slice of the 3D state volume.
    """

    from scipy import ndimage

    crop = np.asarray(crop_slice) > 0
    output = np.zeros(tuple(original_shape_xy), dtype=np.uint8)
    originals_x = (
        (np.arange(original_shape_xy[0]) - float(center_xy[0]))
        * float(original_spacing_xy[0])
    )
    originals_y = (
        (np.arange(original_shape_xy[1]) - float(center_xy[1]))
        * float(original_spacing_xy[1])
    )
    # Crop-space coordinate of each original-grid pixel, from the forward
    # sampling rule x_orig = center + ((i+0.5)/N - 0.5) * field / spacing.
    crop_x = (originals_x / field_mm + 0.5) * output_size - 0.5
    crop_y = (originals_y / field_mm + 0.5) * output_size - 0.5
    grid_x, grid_y = np.meshgrid(crop_x, crop_y, indexing="ij")
    sampled = ndimage.map_coordinates(
        crop.astype(np.float32),
        [grid_x, grid_y],
        order=0,
        mode="constant",
        cval=0.0,
        prefilter=False,
    )
    output[sampled > 0.5] = 1
    return output


def _runtime_components(
    *,
    state: np.ndarray,
    episode_key: str,
    m_sha256: str,
    spacing_xyz: np.ndarray,
    coordinates: np.ndarray,
    prompted_z: int,
    crop: Mapping[str, Any],
    operation: str,
) -> dict[str, Any]:
    """Runtime component enumeration + cue-hit binding, mirroring the frozen
    data-lane candidate materializer without touching its partition guard."""

    enumeration = enumerate_components(
        state,
        episode_id=episode_key,
        m_sha256=m_sha256,
        spacing_xyz=np.asarray(spacing_xyz, dtype=np.float64),
        cue_voxels=coordinates,
        prompted_z=prompted_z,
    )
    components: list[dict[str, Any]] = []
    central_masks: list[np.ndarray] = []
    for component in enumeration.components:
        if component.prompted_slice_mask is None:
            raise V3FullChainError("runtime enumeration omitted prompted slice mask")
        central_mask = physical_crop_resample_2d(
            component.prompted_slice_mask,
            center_xy=crop["center_xy"],
            spacing_xy=np.asarray(spacing_xyz[:2], dtype=np.float32),
            field_mm=float(crop["field_mm"]),
            output_size=int(crop["output_size"]),
            order=0,
        )
        central_mask = (np.asarray(central_mask) > 0.5).astype(np.uint8)
        record = component.as_dict()
        record["candidate_position"] = int(component.position)
        record["descriptor_vector"] = component.descriptor_vector()
        record["prompted_slice_mask"] = central_mask.tolist()
        components.append(record)
        central_masks.append(central_mask)
    cue_hit_position = None
    if operation == "REMOVE":
        cue_hit_position = cue_hit_component_position(
            enumeration, coordinates, state
        )
        if cue_hit_position is None:
            raise V3FullChainError("REMOVE cue did not hit current-state membership")
    descriptor_vectors = np.zeros(
        (1, max(1, len(components)), COMPONENT_DESCRIPTOR_DIM), dtype=np.float32
    )
    valid = np.zeros((1, max(1, len(components))), dtype=bool)
    for index, component in enumerate(components):
        descriptor_vectors[0, index] = np.asarray(
            component["descriptor_vector"], dtype=np.float32
        )
        valid[0, index] = True
    return {
        "components": components,
        "central_masks": central_masks,
        "cue_hit_position": cue_hit_position,
        "descriptor_vectors": descriptor_vectors,
        "valid": valid,
        "enumeration_version": enumeration.enumeration_version,
    }


def _run_compiler(
    *,
    model: ProgramCompilerNet,
    compiler: LegalCallCompiler,
    visual: np.ndarray,
    spacing_xy: np.ndarray,
    operation: str,
    operation_id: int,
    descriptor_vectors: np.ndarray,
    valid: np.ndarray,
    components: list[dict[str, Any]],
    cue_hit_position: int | None,
    device: torch.device,
) -> dict[str, Any]:
    with torch.no_grad():
        outputs = model(
            torch.from_numpy(visual)[None].to(device),
            torch.from_numpy(spacing_xy)[None].to(device),
            torch.tensor([operation_id], device=device),
            torch.from_numpy(descriptor_vectors).to(device),
            torch.from_numpy(valid).to(device),
        )
    family_logits = outputs["family_logits"][0]
    if operation == "REMOVE" and not bool(model.include_repair):
        family_logits = family_logits[:2]
    pointer_probs = None
    valid_tensor = torch.from_numpy(valid[0]).to(device)
    if bool(valid_tensor.any()):
        pointer_probs = torch.softmax(
            outputs["pointer_logits"][0][valid_tensor], dim=0
        ).cpu()
        if not bool(valid_tensor[: len(components)].all()) or bool(
            valid_tensor[len(components) :].any()
        ):
            raise V3FullChainError("padded candidate mask is not contiguous")
    compiled = compiler(
        operation,
        family_logits.cpu(),
        pointer_probs,
        components,
        cue_hit_component_position=cue_hit_position,
    )
    trace = list(compiled["typed_trace"])
    trace.insert(
        -1,
        {
            "step": "PROTECT",
            "protected_refs": dict(
                protected_refs_policy(operation, str(compiled["call"]["operand"]))
            ),
        },
    )
    # JSON-finite boundary (M-15 class): the typed trace must survive strict
    # serialization before it may enter any immutable artifact.
    json.dumps(trace, ensure_ascii=False, allow_nan=False, sort_keys=True)
    return {
        "call": compiled["call"],
        "typed_trace": trace,
        "confidence": float(torch.softmax(family_logits, dim=0).max().item()),
        "goal": render_goal(operation, str(compiled["call"]["family"])),
    }


def _run_editor(
    *,
    model: ProgramEditorUNet2D,
    visual: np.ndarray,
    m0_slice: np.ndarray,
    selected_mask: np.ndarray,
    family: str,
    operand: str,
    operation_id: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    signed_cue = visual[15:16] - visual[16:17]
    editor_visual = np.concatenate(
        [visual[:10], visual[12:13], signed_cue, selected_mask[None]], axis=0
    ).astype(np.float32)
    family_id = family_to_id(family)
    operand_mode = 0 if operand != NEW_CUE_SENTINEL else 1
    with torch.no_grad():
        logits = model(
            torch.from_numpy(editor_visual)[None].to(device),
            torch.tensor([family_id], device=device),
            torch.tensor([operand_mode], device=device),
            torch.tensor([0], device=device),
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


def run_one_case(
    *,
    source: Mapping[str, Any],
    strategy: str,
    simulator: Any,
    metric_evaluator_class: Any,
    compiler_model: ProgramCompilerNet,
    compiler: LegalCallCompiler,
    editor_model: ProgramEditorUNet2D,
    model_config: Mapping[str, Any],
    output_parent: Path,
    device: torch.device,
) -> dict[str, Any]:
    case_id = str(source["case_id"])
    final_dir = output_parent / "cases" / case_id
    final_receipt = final_dir / "case.json"
    if final_dir.exists() or final_dir.is_symlink():
        receipt = json.loads(
            _regular(final_receipt, label="case receipt").read_text(encoding="utf-8")
        )
        _verify_seal(receipt, "case_sha256", label="case receipt")
        if receipt.get("case_id") != case_id or receipt.get("strategy") != strategy:
            raise V3FullChainError("existing case receipt identity differs")
        return receipt
    work_parent = output_parent / ".work"
    work_parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f"{case_id}.", dir=work_parent))
    try:
        pet_image = nib.load(str(_regular(Path(str(source["pet_path"])), label="PET")))
        ct_image = nib.load(str(_regular(Path(str(source["ct_path"])), label="CT")))
        gt_image = nib.load(str(_regular(Path(str(source["gt_path"])), label="GT")))
        m0_image = nib.load(str(_regular(Path(str(source["m0_path"])), label="M0")))
        shape = gt_image.shape
        for label, image in (("PET", pet_image), ("CT", ct_image), ("M0", m0_image)):
            if image.shape != shape or not np.allclose(image.affine, gt_image.affine):
                raise V3FullChainError(f"{case_id} {label}/GT geometry differs")
        spacing = tuple(float(value) for value in gt_image.header.get_zooms()[:3])
        spacing_xyz = np.asarray(spacing, dtype=np.float64)
        pet = np.asanyarray(pet_image.dataobj)
        ct = np.asanyarray(ct_image.dataobj)
        gt = (np.asanyarray(gt_image.dataobj) > 0).astype(np.uint8)
        state = (np.asanyarray(m0_image.dataobj) > 0).astype(np.uint8)
        state_sha = hashlib.sha256(state.tobytes()).hexdigest()
        foreground: set[tuple[int, int, int]] = set()
        background: set[tuple[int, int, int]] = set()
        states: list[dict[str, Any]] = []
        for state_index in range(int(PROTOCOL["evaluation_states"])):
            correction: dict[str, Any] | None = None
            trace: list[dict[str, Any]] | None = None
            plane_dice: float | None = None
            plane_ceiling_dice: float | None = None
            if state_index > 0:
                if np.any(gt):
                    correction = choose_correction(
                        state, gt, strategy=strategy, simulator=simulator
                    )
                    selected = {
                        tuple(int(value) for value in coordinate)
                        for coordinate in correction["coordinates"]
                    }
                    if correction["polarity"] == "foreground":
                        foreground.update(selected)
                    else:
                        background.update(selected)
                else:
                    correction = {
                        "polarity": "none-empty-gt",
                        "coordinates": [],
                        "selected_size": 0,
                        "background_candidate_size": None,
                        "foreground_candidate_size": None,
                    }
            if correction is not None and correction.get("coordinates"):
                coordinates = np.unique(
                    np.asarray(
                        [[int(v) for v in c] for c in correction["coordinates"]],
                        dtype=np.int64,
                    ),
                    axis=0,
                )
                z_values = np.unique(coordinates[:, 2])
                center_z = int(np.bincount(np.searchsorted(z_values, coordinates[:, 2])).argmax())
                center_z = int(z_values[center_z])
                if len(z_values) != 1:
                    # Official scribble spans slices; the 2D bridge uses the
                    # mode slice and records the fact (policy flag).
                    pass
                operation = "ADD" if correction["polarity"] == "foreground" else "REMOVE"
                plane_coords = coordinates[coordinates[:, 2] == center_z]
                mixed_polarity = bool(
                    (operation == "ADD" and background) or (operation == "REMOVE" and foreground)
                )
                episode_key = f"test-{case_id}-state-{state_index}"
                crop = _runtime_crop(
                    pet=pet,
                    ct=ct,
                    state=state,
                    coordinates=plane_coords,
                    center_z=center_z,
                    field_mm=float(model_config["field_mm"]),
                    output_size=int(model_config["output_size"]),
                    original_spacing_xy=np.asarray(
                        gt_image.header.get_zooms()[:2], dtype=np.float32
                    ),
                )
                crop["field_mm"] = float(model_config["field_mm"])
                crop["output_size"] = int(model_config["output_size"])
                components_bundle = _runtime_components(
                    state=state,
                    episode_key=episode_key,
                    m_sha256=state_sha,
                    spacing_xyz=spacing_xyz,
                    coordinates=plane_coords,
                    prompted_z=center_z,
                    crop=crop,
                    operation=operation,
                )
                visual_17 = crop["visual"]
                visual_17[15:16] = crop["scribble_slice"] if operation == "ADD" else 0
                visual_17[16:17] = crop["scribble_slice"] if operation == "REMOVE" else 0
                compiled = _run_compiler(
                    model=compiler_model,
                    compiler=compiler,
                    visual=visual_17,
                    spacing_xy=np.asarray([model_config["expected_spacing"]] * 2, dtype=np.float32),
                    operation=operation,
                    operation_id=0 if operation == "ADD" else 1,
                    descriptor_vectors=components_bundle["descriptor_vectors"],
                    valid=components_bundle["valid"],
                    components=components_bundle["components"],
                    cue_hit_position=components_bundle["cue_hit_position"],
                    device=device,
                )
                call = compiled["call"]
                selected_mask = np.zeros_like(crop["m0_slice"], dtype=np.float32)
                operand = str(call["operand"])
                if operation == "ADD" and operand != NEW_CUE_SENTINEL:
                    operand_index = int(
                        next(
                            index
                            for index, component in enumerate(
                                components_bundle["components"]
                            )
                            if component["component_key"] == operand
                        )
                    )
                    selected_mask = components_bundle["central_masks"][operand_index]
                elif operation == "REMOVE":
                    selected_mask = components_bundle["central_masks"][
                        int(components_bundle["cue_hit_position"])
                    ]
                delta_slice, corrected_slice = _run_editor(
                    model=editor_model,
                    visual=visual_17,
                    m0_slice=crop["m0_slice"],
                    selected_mask=selected_mask,
                    family=str(call["family"]),
                    operand=operand,
                    operation_id=0 if operation == "ADD" else 1,
                    device=device,
                )
                state = state.copy()
                state[:, :, center_z] = _restore_crop_slice_to_original(
                    corrected_slice,
                    center_xy=crop["center_xy"],
                    original_spacing_xy=np.asarray(
                        gt_image.header.get_zooms()[:2], dtype=np.float32
                    ),
                    field_mm=float(model_config["field_mm"]),
                    output_size=int(model_config["output_size"]),
                    original_shape_xy=state.shape[:2],
                )
                state_sha = hashlib.sha256(state.tobytes()).hexdigest()
                trace = compiled["typed_trace"]
                plane_dice = _dice(state[:, :, center_z], gt[:, :, center_z])
                ceiling_state = state.copy()
                ceiling_state[:, :, center_z] = gt[:, :, center_z]
                plane_ceiling_dice = _dice(ceiling_state, gt)
                _write_json_exclusive(
                    temporary / f"typed_trace_state_{state_index}.json",
                    {
                        "case_id": case_id,
                        "state_index": state_index,
                        "typed_trace": trace,
                        "goal": compiled["goal"],
                        "confidence": compiled["confidence"],
                        "grammar_version": GRAMMAR_VERSION,
                        "cue_policy": CUE_POLICY,
                        "plane_policy": PLANE_POLICY,
                        "mixed_polarity_round": mixed_polarity,
                        "accumulated_foreground_voxels": len(foreground),
                        "accumulated_background_voxels": len(background),
                        "scribble_z_values": z_values.tolist(),
                        "prompted_z": center_z,
                    },
                )
            metrics = compute_state_metrics(
                state,
                gt,
                spacing=spacing,
                metric_evaluator_class=metric_evaluator_class,
                case_name=f"v3fullchain-{case_id}-state-{state_index}",
            )
            state_payload = {
                "state": state_index,
                "correction": correction,
                "foreground_scribble_voxels": len(foreground),
                "background_scribble_voxels": len(background),
                "trace_recorded": trace is not None,
                "plane_dice_2d_prompted": plane_dice,
                "single_slice_ceiling_dice_3d": plane_ceiling_dice,
                **metrics,
            }
            if state_index == 0:
                state_payload["plane_dice_2d_prompted"] = None
                state_payload["single_slice_ceiling_dice_3d"] = None
            states.append(state_payload)
        payload = _seal(
            {
                "schema_version": CASE_SCHEMA,
                "case_id": case_id,
                "patient_id": str(source["patient_id"]),
                "data_scope": str(source.get("data_scope") or "AUTHORIZED_LOCKED_TEST"),
                "strategy": strategy,
                "protocol": {
                    **PROTOCOL,
                    "evaluation_states": int(PROTOCOL["evaluation_states"]),
                    "correction_rounds": int(PROTOCOL["evaluation_states"]) - 1,
                },
                "bridge_protocol": PENDING_AMENDMENT_FIELDS,
                "source": {
                    "ct": _file_record(Path(str(source["ct_path"])), label="CT"),
                    "pet": _file_record(Path(str(source["pet_path"])), label="PET"),
                    "gt": _file_record(Path(str(source["gt_path"])), label="GT"),
                    "m0": _file_record(Path(str(source["m0_path"])), label="M0"),
                },
                "states": states,
                "auc": {
                    "auc_dice_3d": _trapz([state["dice"] for state in states]),
                    "auc_dmm_3d": _trapz([state["dmm_f1"] for state in states]),
                    # State 0 has no prompted plane; the 2D supplementary
                    # domain is reported as a rounds-1..5 mean, never as a
                    # six-state AUC and never merged with the 3D domain.
                    "mean_plane_dice_2d_prompted": _mean(
                        [
                            state["plane_dice_2d_prompted"]
                            for state in states
                            if state["plane_dice_2d_prompted"] is not None
                        ]
                    ),
                },
            },
            "case_sha256",
        )
        _write_json_exclusive(temporary / "case.json", payload)
        final_dir.parent.mkdir(parents=True, exist_ok=True)
        os.replace(temporary, final_dir)
        return payload
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _trapz(values: Sequence[float | None]) -> float | None:
    if len(values) != int(PROTOCOL["evaluation_states"]):
        raise V3FullChainError("formal per-case AUC requires exactly six states")
    if any(value is None for value in values):
        return None
    return float(np.trapz(np.asarray(values, dtype=float), np.arange(len(values))))


def _publish_arm_summary(
    *, output: Path, cases: list[dict[str, Any]], receipt_sha256: str
) -> dict[str, Any]:
    eligible = [case for case in cases if case["auc"]["auc_dice_3d"] is not None]
    empty = [case for case in cases if case["auc"]["auc_dice_3d"] is None]
    payload = _seal(
        {
            "schema_version": ARM_SCHEMA,
            "test_access_receipt_sha256": receipt_sha256,
            "protocol": PROTOCOL,
            "bridge_protocol": PENDING_AMENDMENT_FIELDS,
            "case_count": len(cases),
            "eligible_positive_gt_case_count": len(eligible),
            "empty_gt_case_count": len(empty),
            "mean_auc_dice_3d": _mean([case["auc"]["auc_dice_3d"] for case in eligible]),
            "mean_auc_dmm_3d": _mean([case["auc"]["auc_dmm_3d"] for case in eligible]),
            "mean_plane_dice_2d_prompted_supplementary": _mean(
                [case["auc"]["mean_plane_dice_2d_prompted"] for case in eligible]
            ),
            "mean_final_dice_3d": _mean(
                [case["states"][-1]["dice"] for case in eligible]
            ),
            "mean_final_dmm_3d": _mean(
                [case["states"][-1]["dmm_f1"] for case in eligible]
            ),
            "case_receipts": [
                {"case_id": case["case_id"], "case_sha256": case["case_sha256"]}
                for case in cases
            ],
        },
        "arm_sha256",
    )
    _write_json_exclusive(output / "ARM_SUMMARY.json", payload)
    return payload


def build_synthetic_smoke_models(output: Path) -> tuple[Path, Path]:
    """Deterministic tiny compiler/editor checkpoints for contract smoke only."""

    torch.manual_seed(3407)
    compiler_model = ProgramCompilerNet(include_repair=True)
    editor_model = ProgramEditorUNet2D(conditioner="program")
    output.mkdir(parents=True, exist_ok=True)
    compiler_path = output / "synthetic_compiler.pt"
    editor_path = output / "synthetic_editor.pt"
    torch.save(
        {
            "schema_version": COMPILER_CHECKPOINT_SCHEMA,
            "architecture_id": "matched_legal_component_program_v1",
            "hyperparameters": {"include_repair": True},
            "source_m0_lineage": MAINLINE_SOURCE,
            "lineage_receipt_sha256": "synthetic-smoke",
            "episodes_sha256": "synthetic-smoke",
            "state_dict": {
                key: value.detach().cpu().clone()
                for key, value in compiler_model.state_dict().items()
            },
        },
        compiler_path,
    )
    torch.save(
        {
            "schema_version": EDITOR_CHECKPOINT_SCHEMA,
            "architecture_id": "matched_legal_component_program_editor_v1",
            "arm": "J9",
            "call_source": "gold",
            "state_dict": {
                key: value.detach().cpu().clone()
                for key, value in editor_model.state_dict().items()
            },
        },
        editor_path,
    )
    return compiler_path, editor_path


def formal_run(
    *,
    receipt_path: Path,
    experiment_config: Path,
    learning_split: Path,
    run_root: Path,
    case_manifest: Path,
    compiler_checkpoint: Path,
    lineage_receipt: Path,
    editor_checkpoint: Path,
    strategy: str,
    output_label: str,
    official_simulator: Path,
    official_metrics: Path,
    model_config: Mapping[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    """Formal run: exactly-once test access, sealed per-case receipts, summary."""

    if strategy not in ("centerline", "random", "boundary"):
        raise V3FullChainError("strategy must be centerline/random/boundary")
    output = Path(run_root) / f"v3_fullchain_{output_label}"
    if output.is_symlink():
        raise V3FullChainError("arm output must not be a symlink")
    enforce_partition_access(
        partitions={"test"},
        receipt_path=receipt_path,
        experiment_config=experiment_config,
        learning_split=learning_split,
        run_root=run_root,
        output_paths=[output],
    )
    receipt = json.loads(
        _regular(receipt_path, label="test receipt").read_text(encoding="utf-8")
    )
    simulator_path, metrics_path = _validate_pinned_official_smoke_code(
        official_simulator, official_metrics
    )
    summary_path = output / "ARM_SUMMARY.json"
    if summary_path.exists():
        summary = json.loads(
            _regular(summary_path, label="arm summary").read_text(encoding="utf-8")
        )
        _verify_seal(summary, "arm_sha256", label="arm summary")
        return summary
    output.mkdir(parents=True, exist_ok=False)
    rows = [
        _verified_case_row(json.loads(line))
        for line in _regular(case_manifest, label="case manifest")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    if not rows:
        raise V3FullChainError("case manifest is empty")
    case_ids = [str(row["case_id"]) for row in rows]
    if len(set(case_ids)) != len(case_ids):
        raise V3FullChainError("case manifest has duplicate case_id")
    simulator = _load_module(simulator_path, "v3_official_simulator")
    metrics = _load_module(metrics_path, "v3_official_metrics")
    compiler_model, compiler, compiler_ckpt = _load_compiler_bundle(
        compiler_checkpoint, lineage_receipt, device
    )
    editor_model, editor_ckpt = _load_editor_bundle(editor_checkpoint, device)
    audit = {
        "compiler_checkpoint": _file_record(compiler_checkpoint, label="compiler checkpoint"),
        "editor_checkpoint": _file_record(editor_checkpoint, label="editor checkpoint"),
        "lineage_receipt": _file_record(lineage_receipt, label="lineage receipt"),
        "compiler_training_episodes_sha256": str(
            compiler_ckpt.get("episodes_sha256")
        ),
        "editor_training_episodes_sha256": str(editor_ckpt.get("episodes_sha256")),
        "editor_arm": str(editor_ckpt.get("arm")),
        "editor_call_source": str(editor_ckpt.get("call_source")),
        "checkpoint_binding_policy": CHECKPOINT_BINDING_POLICY,
    }
    cases = []
    for row in rows:
        cases.append(
            run_one_case(
                source=row,
                strategy=strategy,
                simulator=simulator,
                metric_evaluator_class=metrics.MetricEvaluator,
                compiler_model=compiler_model,
                compiler=compiler,
                editor_model=editor_model,
                model_config=model_config,
                output_parent=output,
                device=device,
            )
        )
    summary = _publish_arm_summary(
        output=output,
        cases=cases,
        receipt_sha256=receipt.get("receipt_sha256", ""),
    )
    summary["audit"] = audit
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-config", type=Path, required=True)
    parser.add_argument("--learning-split", type=Path, required=True)
    parser.add_argument("--case-manifest", type=Path, required=True)
    parser.add_argument("--compiler-checkpoint", type=Path, required=True)
    parser.add_argument("--lineage-receipt", type=Path, required=True)
    parser.add_argument("--editor-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--strategy", choices=("centerline", "random", "boundary"), required=True
    )
    parser.add_argument("--output-label", required=True)
    parser.add_argument(
        "--official-simulator",
        type=Path,
        default=SCRIPTS_ROOT.parent
        / "upstream"
        / "autoPETV"
        / "interactive"
        / "simulate_scribbles.py",
    )
    parser.add_argument(
        "--official-metrics",
        type=Path,
        default=SCRIPTS_ROOT.parent / "upstream" / "autoPETV" / "metrics.py",
    )
    parser.add_argument("--field-mm", type=float, default=64.0)
    parser.add_argument("--output-size", type=int, default=128)
    parser.add_argument("--expected-spacing", type=float, default=1.0)
    parser.add_argument("--device", default="cuda")
    add_leaf_test_access_arguments(parser)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.run_root is None:
        parser.error("formal run requires --run-root bound by the test receipt")
    if args.test_access_receipt is None:
        parser.error("formal run requires --test-access-receipt")
    try:
        summary = formal_run(
            receipt_path=args.test_access_receipt,
            experiment_config=args.experiment_config,
            learning_split=args.learning_split,
            run_root=args.run_root,
            case_manifest=args.case_manifest,
            compiler_checkpoint=args.compiler_checkpoint,
            lineage_receipt=args.lineage_receipt,
            editor_checkpoint=args.editor_checkpoint,
            strategy=args.strategy,
            output_label=args.output_label,
            official_simulator=args.official_simulator,
            official_metrics=args.official_metrics,
            model_config={
                "field_mm": args.field_mm,
                "output_size": args.output_size,
                "expected_spacing": args.expected_spacing,
            },
            device=torch.device(args.device),
        )
    except (V3FullChainError, TestAccessError) as exc:
        parser.error(str(exc))
    print(json.dumps(
        {key: summary[key] for key in ("case_count", "mean_auc_dice_3d", "mean_auc_dmm_3d")},
        sort_keys=True,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
