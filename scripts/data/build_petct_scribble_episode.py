#!/usr/bin/env python3
"""Build one auditable ADD/REMOVE PET/CT residual-cue episode.

The generator deliberately reuses the pinned autoPET V residual simulator via
an injected callable.  It does not reimplement the scribble algorithms, run the
official FG/BG competition, or expose GT-derived truth to the visible packet.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np
from scipy import ndimage

SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))
for support_dir in (SCRIPTS_ROOT / "baseline", SCRIPTS_ROOT / "data"):
    if str(support_dir) not in sys.path:
        sys.path.insert(0, str(support_dir))


STRATEGIES = ("centerline", "random", "boundary")
from common.petct_route_a_core import (  # noqa: E402
    LEGAL_JOINT_GOALS,
    intent_slots_from_goal,
)


GOALS = LEGAL_JOINT_GOALS
INTENT_SCHEMA_VERSION = "PETCT-INTENT-v2.0"
VISIBLE_SCHEMA_VERSION = "PETCT-EPISODE-VISIBLE-v2.0"
EVAL_SCHEMA_VERSION = "PETCT-EPISODE-EVAL-v2.0"
DEFAULT_UPSTREAM_COMMIT = "4a2026866bfacc812492cfc7e6a8c54ac3c4f703"
DEFAULT_SIMULATOR_SHA256 = (
    "a2124e8aa4207e53ac93259214a35b7cf74626f83ab164e519769f86557d7cd2"
)
DEFAULT_RUNTIME_MANIFEST = (
    SCRIPTS_ROOT.parent / "protocols" / "autopetv_protocol_runtime.json"
)
DEFAULT_RUNTIME_MANIFEST_SHA256 = (
    "100e4e6453dbf88d3aebc8b4c8107f574c35c08119a1d6789f9fd166d13c4854"
)
AUTOPETV_RUNTIME_SCHEMA = "PETCT-AUTOPETV-PROTOCOL-RUNTIME-v2.0"
AUTOPETV_RUNTIME_STATUS = (
    "FROZEN_MINIMAL_RUNTIME_SIX_CLASS_POLARITY_ADAPTER_NOT_EXECUTED"
)
AUTOPETV_RUNTIME_LICENSE = "Apache-2.0"
AUTOPETV_RUNTIME_REPOSITORY = "https://github.com/lab-midas/autoPETV"
AUTOPETV_RUNTIME_ALLOWLIST = (
    "LICENSE",
    "interactive/simulate_scribbles.py",
    "metrics.py",
)
FORBIDDEN_VISIBLE_KEY_FRAGMENTS = (
    "gt",
    "gold",
    "goal",
    "label",
    "eval",
    "evaluation",
    "residual",
    "component",
    "authorized",
    "target",
    "source_case",
    "source_patient",
)
FORBIDDEN_VISIBLE_VALUE_FRAGMENTS = (
    "gold",
    "goal",
    "label",
    "eval",
    "evaluation",
    "authorized",
    "target",
    "source_case",
    "source_patient",
)
FORBIDDEN_VISIBLE_GOAL_VALUES = frozenset(
    str(goal).casefold() for goal in LEGAL_JOINT_GOALS
)

# Visible provenance is projected from the complete audit provenance.  Unknown
# fields are never copied: the audit/evaluation document retains the complete
# object, while the inference-visible document receives only these fields.
CONTROLLED_VISIBLE_M0_PROVENANCE_KEYS = frozenset(
    {
        "kind",
        "schema_version",
        "operation",
        "state_content_sha256",
        "cue_coordinate_sha256",
    }
)
NATURAL_VISIBLE_M0_PROVENANCE_KEYS = frozenset(
    {
        "kind",
        "schema_version",
        "contract_version",
        "m0_sha256",
        "foreground_probability_sha256",
        "checkpoint_sha256",
        "plans_sha256",
        "dataset_json_sha256",
        "source_tree_sha256",
        "splits_final_sha256",
        "preprocess_ready_sha256",
        "full_train_ready_sha256",
        "fold_receipt_sha256",
        "input_ct_sha256",
        "input_pet_sha256",
    }
)
NATURAL_PROVENANCE_HASH_KEYS = (
    "oof_ready_sha256",
    "m0_sha256",
    "foreground_probability_sha256",
    "checkpoint_sha256",
    "plans_sha256",
    "dataset_json_sha256",
    "source_tree_sha256",
    "splits_final_sha256",
    "preprocess_ready_sha256",
    "full_train_ready_sha256",
    "fold_receipt_sha256",
    "input_ct_sha256",
    "input_pet_sha256",
    "input_gt_sha256",
)
# The v6 OOF has no canonical files for the legacy training-receipt hashes, so
# the binding expresses their absence as explicit null (auditable absence, see
# M-17).  A v5-era binding must still carry real digests for these three keys.
NATURAL_PROVENANCE_NULLABLE_KEYS = frozenset(
    {
        "preprocess_ready_sha256",
        "full_train_ready_sha256",
        "fold_receipt_sha256",
    }
)


RESIDUAL_COMPONENT_CONNECTIVITY_18 = ndimage.generate_binary_structure(3, 2)
DEFAULT_MINIMUM_BEST_SLICE_PIXELS = 1
CUE_ELIGIBILITY_RULE = "largest-axial-slice-pixels-per-residual-component"
CUE_INELIGIBLE_REASON = "INELIGIBLE_BEST_SLICE_BELOW_MIN_PIXELS"


class EpisodeContractError(RuntimeError):
    """Raised when an episode would violate the frozen data contract."""


class ResidualCueIneligibleError(EpisodeContractError):
    """Raised when no residual component can physically carry a scribble cue.

    This is a *counted, reasoned* exclusion, not a contract breach: a residual
    fragment that occupies a handful of pixels on its widest axial slice cannot
    carry a centerline or a boundary curve at all.  It stays a subclass of
    ``EpisodeContractError`` so existing callers keep failing closed unless they
    opt in to counting the exclusion.
    """

    def __init__(self, message: str, *, census: dict[str, Any]) -> None:
        super().__init__(message)
        self.census = dict(census)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_json(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return _sha256_bytes(encoded)


def _mask_sha256(mask: np.ndarray) -> str:
    binary = np.asarray(mask, dtype=np.bool_)
    header = json.dumps(
        {"shape": list(binary.shape), "axis_order": "xyz"},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return _sha256_bytes(header + np.packbits(binary.reshape(-1)).tobytes())


def _binary_3d(mask: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(mask)
    if array.ndim != 3:
        raise EpisodeContractError(f"{name} must be a 3D binary mask")
    if not np.issubdtype(array.dtype, np.bool_):
        if not np.issubdtype(array.dtype, np.number):
            raise EpisodeContractError(f"{name} must be binary")
        if not np.all(np.isfinite(array)) or not np.all(np.isin(array, [0, 1])):
            raise EpisodeContractError(f"{name} must be binary with values 0/1")
    return array.astype(np.bool_, copy=False)


def compute_fn_residual(gt: np.ndarray, m0: np.ndarray) -> np.ndarray:
    r"""Return ``G \ M0`` after strict binary/shape validation."""
    gt_binary = _binary_3d(gt, name="GT")
    m0_binary = _binary_3d(m0, name="M0")
    if gt_binary.shape != m0_binary.shape:
        raise EpisodeContractError(
            f"GT/M0 shape mismatch: {gt_binary.shape} != {m0_binary.shape}"
        )
    return gt_binary & ~m0_binary


def compute_fp_residual(gt: np.ndarray, m0: np.ndarray) -> np.ndarray:
    r"""Return ``M0 \ G`` after strict binary/shape validation."""

    gt_binary = _binary_3d(gt, name="GT")
    m0_binary = _binary_3d(m0, name="M0")
    if gt_binary.shape != m0_binary.shape:
        raise EpisodeContractError(
            f"GT/M0 shape mismatch: {gt_binary.shape} != {m0_binary.shape}"
        )
    return m0_binary & ~gt_binary


def assign_scribble_strategy(patient_id: str, *, salt: str) -> str:
    """Assign exactly one primary strategy deterministically at patient level."""
    if not patient_id or not salt:
        raise EpisodeContractError("patient_id and strategy salt must be non-empty")
    digest = hashlib.sha256(f"{salt}|{patient_id}".encode("utf-8")).hexdigest()
    return STRATEGIES[int(digest, 16) % len(STRATEGIES)]


def validated_minimum_best_slice_pixels(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise EpisodeContractError("minimum_best_slice_pixels must be an integer")
    threshold = int(value)
    if threshold < 1:
        raise EpisodeContractError("minimum_best_slice_pixels must be at least 1")
    return threshold


def residual_component_census(
    residual: np.ndarray, *, minimum_best_slice_pixels: int
) -> tuple[np.ndarray, dict[str, Any]]:
    r"""Split a residual into the components that can actually carry a cue.

    A residual connected component is eligible only when its largest axial
    slice holds at least ``minimum_best_slice_pixels`` pixels.  The unit is
    pixels rather than mm\ :sup:`2` on purpose: in-plane spacing varies per case
    (4.073 mm and 2.734 mm both occur), so a single pixel spans anywhere from
    7.48 mm\ :sup:`2` to 16.59 mm\ :sup:`2` and no physical-area threshold can
    guarantee that a centerline or boundary curve is drawable.

    Returns the union of eligible components plus an auditable census.  The
    caller decides whether an empty union is an exclusion or a hard failure.
    """

    threshold = validated_minimum_best_slice_pixels(minimum_best_slice_pixels)
    binary = _binary_3d(residual, name="residual")
    if threshold <= 1:
        # Every non-empty component clears a one-pixel floor, so the labelling
        # pass is pure cost and the eligible union is the residual itself.
        return binary, {
            "rule": CUE_ELIGIBILITY_RULE,
            "minimum_best_slice_pixels": threshold,
            "component_connectivity": 18,
            "enforced": False,
            "component_total": None,
            "component_eligible": None,
            "component_excluded": None,
            "max_excluded_best_slice_pixels": None,
            "eligible_voxels": int(binary.sum()),
            "residual_voxels": int(binary.sum()),
        }
    labels, count = ndimage.label(
        binary, structure=RESIDUAL_COMPONENT_CONNECTIVITY_18
    )
    best_slice_pixels = np.zeros(count + 1, dtype=np.int64)
    for z_index in range(binary.shape[2]):
        slice_labels = labels[:, :, z_index]
        if not slice_labels.any():
            continue
        np.maximum(
            best_slice_pixels,
            np.bincount(slice_labels.ravel(), minlength=count + 1),
            out=best_slice_pixels,
        )
    best_slice_pixels[0] = 0
    keep = best_slice_pixels >= threshold
    keep[0] = False
    excluded = best_slice_pixels[1:][~keep[1:]]
    eligible = keep[labels]
    return eligible, {
        "rule": CUE_ELIGIBILITY_RULE,
        "minimum_best_slice_pixels": threshold,
        "component_connectivity": 18,
        "enforced": True,
        "component_total": int(count),
        "component_eligible": int(keep.sum()),
        "component_excluded": int(count - keep.sum()),
        "max_excluded_best_slice_pixels": (
            int(excluded.max()) if excluded.size else 0
        ),
        "eligible_voxels": int(eligible.sum()),
        "residual_voxels": int(binary.sum()),
    }


def _largest_axial_component(residual: np.ndarray) -> tuple[int, np.ndarray, int]:
    structure = np.ones((3, 3), dtype=np.uint8)
    best_slice = -1
    best_component: np.ndarray | None = None
    best_area = 0
    for z_index in range(residual.shape[2]):
        labels, count = ndimage.label(residual[:, :, z_index], structure=structure)
        if count == 0:
            continue
        areas = np.bincount(labels.reshape(-1))
        areas[0] = 0
        label_id = int(np.argmax(areas))
        area = int(areas[label_id])
        if area > best_area:
            best_slice = z_index
            best_component = labels == label_id
            best_area = area
    if best_component is None:
        raise EpisodeContractError("non-empty operation residual has no selectable component")
    return best_slice, best_component, best_area


def _normalize_coordinates(
    coordinates: Any,
    *,
    residual: np.ndarray,
    source_slice: int,
    source_component: np.ndarray,
) -> list[list[int]]:
    if not isinstance(coordinates, (list, tuple)) or not coordinates:
        raise EpisodeContractError("official simulator returned an empty scribble")
    normalized: list[list[int]] = []
    seen: set[tuple[int, int, int]] = set()
    for raw in coordinates:
        if not isinstance(raw, (list, tuple, np.ndarray)) or len(raw) != 3:
            raise EpisodeContractError(
                "scribble coordinate must contain three xyz indices"
            )
        try:
            coord = tuple(int(value) for value in raw)
        except (TypeError, ValueError) as exc:
            raise EpisodeContractError("scribble coordinate is not integral") from exc
        if any(
            value < 0 or value >= residual.shape[axis]
            for axis, value in enumerate(coord)
        ):
            raise EpisodeContractError(f"scribble coordinate out of bounds: {coord}")
        if not residual[coord]:
            raise EpisodeContractError(
                f"scribble coordinate outside operation residual: {coord}"
            )
        if coord[2] != source_slice or not source_component[coord[0], coord[1]]:
            raise EpisodeContractError(
                f"scribble coordinate outside selected source component: {coord}"
            )
        if coord in seen:
            raise EpisodeContractError(f"duplicate scribble coordinate: {coord}")
        seen.add(coord)
        normalized.append(list(coord))
    return sorted(normalized)


@contextmanager
def _audit_official_strategy_primitives(
    simulator: Callable[..., Any],
) -> Iterator[list[dict[str, str]] | None]:
    """Observe the pinned simulator's real primitive calls without editing it.

    autoPET V catches exceptions around the requested primitive and silently
    calls ``scribble_random``.  The public return value cannot reveal that
    branch, so the adapter temporarily wraps the three module-level primitives,
    records their outcomes, and restores the exact original callables in a
    ``finally`` block.  Injected test doubles have no module binding and remain
    supported, but are explicitly marked as unaudited by the caller.
    """

    module = getattr(simulator, "_petct_official_module", None)
    if module is None:
        yield None
        return
    originals: dict[str, Callable[..., Any]] = {}
    events: list[dict[str, str]] = []
    try:
        for strategy in STRATEGIES:
            name = f"scribble_{strategy}"
            primitive = getattr(module, name, None)
            if not callable(primitive):
                raise EpisodeContractError(
                    f"official simulator module lacks callable {name}"
                )
            originals[name] = primitive

            def audited(*args: Any, _primitive=primitive, _strategy=strategy, **kwargs: Any):
                try:
                    result = _primitive(*args, **kwargs)
                except Exception as exc:
                    events.append(
                        {
                            "strategy": _strategy,
                            "outcome": "RAISED",
                            "exception_type": type(exc).__name__,
                            "exception_message": str(exc)[:512],
                        }
                    )
                    raise
                events.append({"strategy": _strategy, "outcome": "RETURNED"})
                return result

            setattr(module, name, audited)
        yield events
    finally:
        for name, primitive in originals.items():
            setattr(module, name, primitive)


def generate_residual_scribble(
    residual: np.ndarray,
    *,
    operation: str,
    strategy: str,
    simulator: Callable[..., Any],
    upstream_commit: str,
    seed: int = 42,
    minimum_best_slice_pixels: int = DEFAULT_MINIMUM_BEST_SLICE_PIXELS,
) -> dict[str, Any]:
    """Call the official simulator on an FN (ADD) or FP (REMOVE) mask."""
    if operation not in {"ADD", "REMOVE"}:
        raise EpisodeContractError("operation must be ADD or REMOVE")
    residual_name = "FN residual" if operation == "ADD" else "FP residual"
    residual_binary = _binary_3d(residual, name=residual_name)
    if not np.any(residual_binary):
        raise EpisodeContractError(
            f"{operation} episode requires a non-empty {residual_name}"
        )
    if strategy not in STRATEGIES:
        raise EpisodeContractError(
            f"strategy must be one of {STRATEGIES}, received {strategy!r}"
        )
    if not upstream_commit:
        raise EpisodeContractError("upstream_commit must be recorded")

    # Eligibility is decided BEFORE the simulator runs.  Handing the pinned
    # simulator a cloud of pixel-scale fragments makes it select something no
    # strategy can draw on; excluding those candidates up front is a counted
    # exclusion, whereas discovering it afterwards is an unrecoverable error.
    threshold = validated_minimum_best_slice_pixels(minimum_best_slice_pixels)
    eligible_residual, cue_eligibility = residual_component_census(
        residual_binary, minimum_best_slice_pixels=threshold
    )
    if not eligible_residual.any():
        raise ResidualCueIneligibleError(
            "no %s component reaches %d pixels on any axial slice"
            % (residual_name, threshold),
            census=cue_eligibility,
        )
    source_slice, source_component, source_area = _largest_axial_component(
        eligible_residual
    )
    if source_area < threshold:
        # An eligible 3D component can still be split in-plane, so the widest
        # 2D component the simulator will actually draw on is re-checked here.
        raise ResidualCueIneligibleError(
            "widest in-plane %s candidate holds %d of the required %d pixels"
            % (residual_name, source_area, threshold),
            census={**cue_eligibility, "selected_component_area": int(source_area)},
        )
    try:
        with _audit_official_strategy_primitives(simulator) as primitive_events:
            output = simulator(
                eligible_residual.astype(np.uint8), strategy=strategy, seed=int(seed)
            )
    except Exception as exc:
        raise EpisodeContractError(f"official simulator failed: {exc}") from exc
    if not isinstance(output, (tuple, list)) or len(output) != 3:
        raise EpisodeContractError(
            "official simulator must return three values: coordinates, class, size"
        )
    coordinates_raw, label_class, declared_size = output
    if not bool(label_class):
        raise EpisodeContractError("official residual simulator returned an empty class")

    coordinates = _normalize_coordinates(
        coordinates_raw,
        residual=eligible_residual,
        source_slice=source_slice,
        source_component=source_component,
    )
    try:
        size = int(declared_size)
    except (TypeError, ValueError) as exc:
        raise EpisodeContractError("official scribble size is not integral") from exc
    if size != len(coordinates):
        raise EpisodeContractError(
            f"official scribble size mismatch: {size} != {len(coordinates)}"
        )

    source_coordinates = {
        (int(x), int(y), source_slice) for x, y in np.argwhere(source_component)
    }
    coordinate_set = {tuple(coord) for coord in coordinates}
    dense_source_component = coordinate_set == source_coordinates
    density_mode = "dense_source_component" if dense_source_component else "sparse"

    if primitive_events is None:
        effective_strategy = strategy
        strategy_fallback = False
        fallback_reason = None
        strategy_audit = "INJECTED_SIMULATOR_UNAUDITED"
    else:
        returned = [
            event["strategy"]
            for event in primitive_events
            if event.get("outcome") == "RETURNED"
        ]
        if not returned:
            raise EpisodeContractError(
                "official simulator returned without a successful audited primitive call"
            )
        effective_strategy = returned[-1]
        strategy_fallback = effective_strategy != strategy
        strategy_audit = "OFFICIAL_PRIMITIVE_CALL_AUDITED"
        if strategy_fallback:
            failures = [
                event
                for event in primitive_events
                if event.get("strategy") == strategy
                and event.get("outcome") == "RAISED"
            ]
            failure = failures[-1] if failures else None
            fallback_reason = (
                "UPSTREAM_EXCEPTION_FALLBACK_TO_RANDOM"
                if failure is None
                else "UPSTREAM_%s_FALLBACK_TO_RANDOM:%s:%s"
                % (
                    strategy.upper(),
                    failure.get("exception_type", "Exception"),
                    failure.get("exception_message", ""),
                )
            )
        else:
            fallback_reason = None

    return {
        "contract_version": "PETCT-RESIDUAL-CUE-v2.0",
        "upstream": "lab-midas/autoPETV",
        "upstream_commit": upstream_commit,
        "simulator_entrypoint": "interactive.simulate_scribbles.simulate_scribble_from_label",
        "operation": operation,
        "residual_kind": "FN" if operation == "ADD" else "FP",
        "polarity": "foreground" if operation == "ADD" else "background",
        # ``strategy`` remains as a compatibility alias for the requested arm.
        "strategy": strategy,
        "requested_strategy": strategy,
        "effective_strategy": effective_strategy,
        "strategy_fallback": strategy_fallback,
        "fallback_reason": fallback_reason,
        "strategy_audit": strategy_audit,
        "primitive_call_trace": primitive_events or [],
        "seed": int(seed),
        "coordinates_xyz": coordinates,
        "coordinate_count": len(coordinates),
        "coordinate_sha256": _sha256_json(coordinates),
        "residual_sha256": _mask_sha256(residual_binary),
        "residual_voxels": int(residual_binary.sum()),
        # The mining residual above stays the full FN/FP set; the mask below is
        # what the official simulator was actually allowed to see.
        "cue_eligibility": cue_eligibility,
        "eligible_residual_sha256": _mask_sha256(eligible_residual),
        "source_slice": source_slice,
        "source_component_area": source_area,
        "scribble_density_mode": density_mode,
        # Deprecated compatibility alias: this describes density, not strategy.
        "fallback_mode": density_mode,
        "sparse_scribble": not dense_source_component,
    }


def canonical_intent_frame(goal: str) -> dict[str, Any]:
    """Render the frozen deterministic six-class PETCT-INTENT-v2.0 frame."""
    templates = {
        "ADD_SAME_LOCAL": (
            "Add only the missed foreground immediately around the positive cue; "
            "it is same to an existing lesion."
        ),
        "REMOVE_SAME_LOCAL": (
            "Remove only the false-positive foreground immediately around the "
            "negative cue while preserving the same true lesion."
        ),
        "ADD_SAME_COMPLETE": (
            "Complete the missed foreground same to the existing lesion, "
            "including its full residual extent."
        ),
        "REMOVE_SAME_COMPLETE": (
            "Remove the complete false-positive extent same to the retained "
            "true lesion."
        ),
        "ADD_NEW_COMPLETE": (
            "Add the complete positively cued new lesion component."
        ),
        "REMOVE_NEW_COMPLETE": (
            "Remove the complete negatively cued new false-positive component."
        ),
    }
    if goal not in templates:
        raise EpisodeContractError(f"goal must be one of {GOALS}, received {goal!r}")
    try:
        operation, target, scope = intent_slots_from_goal(goal)
    except Exception as exc:
        raise EpisodeContractError(str(exc)) from exc
    return {
        "schema_version": INTENT_SCHEMA_VERSION,
        "decision": "PREDICT",
        "goal": goal,
        "operation": operation,
        "target": target,
        "scope": scope,
        "preserve": [
            "PRESERVE_UNAUTHORIZED_M0",
            "DO_NOT_CHANGE_OUTSIDE_AUTHORIZED_TARGET",
        ],
        "intent_text": templates[goal],
        "alternatives": [],
        "confidence": None,
    }


def _assert_visible_safe(value: Any, *, path: str = "visible") -> None:
    """Reject label/evaluation material anywhere in a visible packet.

    Checking keys alone is insufficient: a neutral key can still carry a gold
    goal or a goal-derived filename.  Every string value is therefore checked
    as well, which covers nested values, paths, and filenames.
    """

    if isinstance(value, dict):
        for key, child in value.items():
            lowered = str(key).casefold()
            if any(
                fragment in lowered for fragment in FORBIDDEN_VISIBLE_KEY_FRAGMENTS
            ):
                raise EpisodeContractError(
                    f"forbidden evaluation field in visible document at {path}.{key}"
                )
            _assert_visible_safe(child, path=f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _assert_visible_safe(child, path=f"{path}[{index}]")
    elif isinstance(value, (str, Path)):
        lowered = str(value).casefold()
        if (
            any(
                fragment in lowered for fragment in FORBIDDEN_VISIBLE_VALUE_FRAGMENTS
            )
            or any(goal in lowered for goal in FORBIDDEN_VISIBLE_GOAL_VALUES)
        ):
            raise EpisodeContractError(
                f"forbidden evaluation value in visible document at {path}"
            )


def _visible_m0_provenance(
    lane: str, provenance: dict[str, Any]
) -> dict[str, Any]:
    """Project complete audit provenance through a lane-specific allowlist."""

    allowed = (
        CONTROLLED_VISIBLE_M0_PROVENANCE_KEYS
        if lane == "controlled"
        else NATURAL_VISIBLE_M0_PROVENANCE_KEYS
    )
    # Null legacy receipt hashes are the auditable absence of a canonical file
    # (v6 OOF); the visible projection omits absent fields while the evaluation
    # document retains the explicit nulls for the audit trail.
    visible = {
        key: provenance[key]
        for key in allowed
        if key in provenance and provenance[key] is not None
    }
    operation = visible.get("operation")
    if operation is not None and operation not in {"ADD", "REMOVE"}:
        raise EpisodeContractError("visible M0 provenance has invalid operation")
    for key, value in visible.items():
        if key.endswith("_sha256") and (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise EpisodeContractError(
                f"visible M0 provenance has invalid {key}"
            )
    _assert_visible_safe(visible, path="visible.m0_provenance")
    return visible


def _validate_m0_provenance(lane: str, provenance: dict[str, Any]) -> None:
    if not isinstance(provenance, dict):
        raise EpisodeContractError("m0_provenance must be an object")
    if lane == "controlled":
        return
    if provenance.get("kind") != "patient_excluded_oof":
        raise EpisodeContractError(
            "natural lane requires patient-excluded OOF M0 provenance"
        )
    fold = provenance.get("held_out_fold")
    if not isinstance(fold, int) or fold not in range(5):
        raise EpisodeContractError("natural OOF held_out_fold must be 0..4")
    for key in NATURAL_PROVENANCE_HASH_KEYS:
        value = provenance.get(key)
        if value is None and key in NATURAL_PROVENANCE_NULLABLE_KEYS:
            # Explicit null is the auditable absence of a legacy training
            # receipt in the v6 OOF lineage; any other null stays invalid.
            continue
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(char not in "0123456789abcdef" for char in value)
        ):
            raise EpisodeContractError(f"natural OOF provenance has invalid {key}")
    binding_hash = provenance.get("binding_sha256")
    unsigned = {
        key: value for key, value in provenance.items() if key != "binding_sha256"
    }
    if binding_hash != _sha256_json(unsigned):
        raise EpisodeContractError("natural OOF provenance binding hash mismatch")


def resolve_m0_provenance(
    *,
    lane: str,
    provenance_json: str | None,
    oof_ready: Path | None,
    case_id: str,
    patient_id: str,
    m0_path: Path,
    natural_validator: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Resolve controlled metadata or enforce the natural OOF trust chain."""
    if lane == "controlled":
        if provenance_json is None:
            raise EpisodeContractError("controlled lane requires --m0-provenance-json")
        try:
            provenance = json.loads(provenance_json)
        except json.JSONDecodeError as exc:
            raise EpisodeContractError("m0-provenance-json is invalid JSON") from exc
        if not isinstance(provenance, dict):
            raise EpisodeContractError("m0-provenance-json must be an object")
        return provenance
    if provenance_json is not None:
        raise EpisodeContractError(
            "natural lane refuses caller-supplied M0 provenance; use --oof-ready"
        )
    if oof_ready is None:
        raise EpisodeContractError("natural lane requires --oof-ready")
    if natural_validator is None:
        from baseline.validate_petct_m0_oof import validate_natural_oof_binding

        natural_validator = validate_natural_oof_binding
    try:
        provenance = natural_validator(
            oof_ready,
            case_id=case_id,
            patient_id=patient_id,
            m0_path=m0_path,
        )
    except Exception as exc:
        raise EpisodeContractError(f"natural OOF M0 gate failed: {exc}") from exc
    _validate_m0_provenance("natural", provenance)
    return provenance


def build_episode_documents(
    *,
    episode_id: str,
    lane: str,
    patient_group_hash: str,
    montage_reference: str,
    m0_provenance: dict[str, Any],
    scribble_record: dict[str, Any],
    source_case_id: str,
    source_patient_id: str,
    residual_sha256: str,
    residual_voxels: int,
    gold_intent: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Create physically separable visible and evaluation documents."""
    if not episode_id or Path(episode_id).name != episode_id:
        raise EpisodeContractError("episode_id must be an opaque filename-safe token")
    if lane not in {"controlled", "natural"}:
        raise EpisodeContractError("lane must be controlled or natural")
    _validate_m0_provenance(lane, m0_provenance)
    if len(patient_group_hash) != 64:
        raise EpisodeContractError("patient_group_hash must be a SHA-256 digest")
    operation = gold_intent.get("operation")
    expected_polarity = {"ADD": "foreground", "REMOVE": "background"}.get(operation)
    if expected_polarity is None or scribble_record.get("polarity") != expected_polarity:
        raise EpisodeContractError("cue polarity must agree with the gold operation")
    if gold_intent.get("schema_version") != INTENT_SCHEMA_VERSION:
        raise EpisodeContractError("gold intent does not match PETCT-INTENT-v2.0")

    scribble_visible_keys = (
        "polarity",
        "strategy",
        "requested_strategy",
        "effective_strategy",
        "strategy_fallback",
        "fallback_reason",
        "strategy_audit",
        "seed",
        "coordinates_xyz",
        "coordinate_count",
        "coordinate_sha256",
        "scribble_density_mode",
        "fallback_mode",
    )
    visible_m0_provenance = _visible_m0_provenance(lane, m0_provenance)
    visible = {
        "schema_version": VISIBLE_SCHEMA_VERSION,
        "episode_id": episode_id,
        "lane": lane,
        "patient_group_hash": patient_group_hash,
        "montage_reference": montage_reference,
        "m0_provenance": visible_m0_provenance,
        "scribble": {
            key: scribble_record[key]
            for key in scribble_visible_keys
            if key in scribble_record
        },
        "expected_model_output_schema": INTENT_SCHEMA_VERSION,
    }
    _assert_visible_safe(visible)

    evaluation = {
        "schema_version": EVAL_SCHEMA_VERSION,
        "episode_id": episode_id,
        "lane": lane,
        "source_case_id": source_case_id,
        "source_patient_id": source_patient_id,
        "residual_sha256": residual_sha256,
        "residual_voxels": int(residual_voxels),
        "gold_intent": gold_intent,
        "scribble_provenance": scribble_record,
        "m0_provenance": m0_provenance,
    }
    return visible, evaluation


def _is_same_or_nested(first: Path, second: Path) -> bool:
    if first == second:
        return True
    try:
        first.relative_to(second)
        return True
    except ValueError:
        pass
    try:
        second.relative_to(first)
        return True
    except ValueError:
        return False


def _json_bytes(payload: dict[str, Any]) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def publish_episode_documents(
    visible: dict[str, Any],
    evaluation: dict[str, Any],
    *,
    visible_root: Path,
    eval_root: Path,
) -> dict[str, Any]:
    """Publish a no-clobber visible/eval pair under physically disjoint roots."""
    visible_root = Path(visible_root).resolve()
    eval_root = Path(eval_root).resolve()
    if _is_same_or_nested(visible_root, eval_root):
        raise EpisodeContractError("visible and eval roots must be physically disjoint")
    if visible.get("episode_id") != evaluation.get("episode_id"):
        raise EpisodeContractError("visible/eval episode_id mismatch")
    episode_id = str(visible.get("episode_id", ""))
    if not episode_id or Path(episode_id).name != episode_id:
        raise EpisodeContractError("invalid episode_id")
    _assert_visible_safe(visible)

    visible_root.mkdir(parents=True, exist_ok=True)
    eval_root.mkdir(parents=True, exist_ok=True)
    visible_path = visible_root / f"{episode_id}.json"
    eval_path = eval_root / f"{episode_id}.json"
    if visible_path.exists() or eval_path.exists():
        raise FileExistsError(
            f"refusing to overwrite existing episode document: {episode_id}"
        )

    visible_bytes = _json_bytes(visible)
    eval_bytes = _json_bytes(evaluation)
    created: list[Path] = []
    try:
        with visible_path.open("xb") as stream:
            created.append(visible_path)
            stream.write(visible_bytes)
            stream.flush()
            os.fsync(stream.fileno())
        with eval_path.open("xb") as stream:
            created.append(eval_path)
            stream.write(eval_bytes)
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        for path in created:
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        raise

    return {
        "status": "COMMITTED",
        "episode_id": episode_id,
        "visible_path": str(visible_path),
        "visible_sha256": _sha256_bytes(visible_bytes),
        "eval_path": str(eval_path),
        "eval_sha256": _sha256_bytes(eval_bytes),
    }


def official_simulator_provenance(
    path: Path,
    *,
    expected_commit: str = DEFAULT_UPSTREAM_COMMIT,
    expected_sha256: str = DEFAULT_SIMULATOR_SHA256,
    runtime_manifest: Path | None = None,
) -> dict[str, Any]:
    raw_path = Path(path)
    if raw_path.is_symlink():
        raise EpisodeContractError("official simulator must not be a symlink")
    path = raw_path.resolve()
    if not path.is_file():
        raise EpisodeContractError(f"official simulator file not found: {path}")
    observed_sha256 = _sha256_bytes(path.read_bytes())
    if observed_sha256 != expected_sha256:
        raise EpisodeContractError("official simulator SHA-256 differs from the pinned blob")
    repo_root = path.parent.parent
    relative = path.relative_to(repo_root).as_posix()
    if relative != "interactive/simulate_scribbles.py":
        raise EpisodeContractError("official simulator path differs from the pinned runtime path")

    if not (repo_root / ".git").exists():
        manifest_path = Path(runtime_manifest or DEFAULT_RUNTIME_MANIFEST)
        if manifest_path.is_symlink() or not manifest_path.is_file():
            raise EpisodeContractError("AutoPET V minimal runtime manifest is missing")
        manifest_path = manifest_path.resolve()
        manifest_bytes = manifest_path.read_bytes()
        manifest_sha256 = _sha256_bytes(manifest_bytes)
        if manifest_sha256 != DEFAULT_RUNTIME_MANIFEST_SHA256:
            raise EpisodeContractError(
                "AutoPET V minimal runtime manifest differs from the pinned manifest"
            )
        try:
            manifest = json.loads(manifest_bytes.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise EpisodeContractError(
                "AutoPET V minimal runtime manifest is invalid UTF-8 JSON"
            ) from exc
        if not isinstance(manifest, dict):
            raise EpisodeContractError("AutoPET V minimal runtime manifest must be an object")
        expected_header = {
            "schema_version": AUTOPETV_RUNTIME_SCHEMA,
            "status": AUTOPETV_RUNTIME_STATUS,
            "upstream_repository": AUTOPETV_RUNTIME_REPOSITORY,
            "upstream_commit": expected_commit,
            "license": AUTOPETV_RUNTIME_LICENSE,
        }
        for key, expected in expected_header.items():
            if manifest.get(key) != expected:
                raise EpisodeContractError(
                    f"AutoPET V minimal runtime manifest {key} mismatch"
                )
        files = manifest.get("files")
        if not isinstance(files, list) or not all(isinstance(item, dict) for item in files):
            raise EpisodeContractError("AutoPET V minimal runtime files must be a list")
        by_path = {str(item.get("path")): item for item in files}
        if len(by_path) != len(files) or tuple(sorted(by_path)) != AUTOPETV_RUNTIME_ALLOWLIST:
            raise EpisodeContractError("AutoPET V minimal runtime allowlist mismatch")
        if by_path["interactive/simulate_scribbles.py"].get("required_callable") != (
            "simulate_scribble_from_label"
        ):
            raise EpisodeContractError("AutoPET V simulator callable contract mismatch")
        if by_path["metrics.py"].get("required_callable") != "MetricEvaluator":
            raise EpisodeContractError("AutoPET V metrics callable contract mismatch")
        if "required_callable" in by_path["LICENSE"]:
            raise EpisodeContractError("AutoPET V LICENSE entry has unexpected callable")
        if by_path[relative].get("sha256") != expected_sha256:
            raise EpisodeContractError("AutoPET V manifest simulator hash mismatch")

        observed_files: set[str] = set()
        observed_dirs: set[str] = set()
        file_records: list[dict[str, Any]] = []
        for candidate in sorted(repo_root.rglob("*")):
            if candidate.is_symlink():
                raise EpisodeContractError(
                    "AutoPET V minimal runtime contains a symlink: "
                    + candidate.relative_to(repo_root).as_posix()
                )
            relative_candidate = candidate.relative_to(repo_root).as_posix()
            if candidate.is_dir():
                observed_dirs.add(relative_candidate)
                continue
            if not candidate.is_file():
                raise EpisodeContractError(
                    "AutoPET V minimal runtime contains a non-regular entry"
                )
            observed_files.add(relative_candidate)
            expected_record = by_path.get(relative_candidate)
            if expected_record is None:
                raise EpisodeContractError(
                    "AutoPET V minimal runtime contains an extra file: "
                    + relative_candidate
                )
            digest = _sha256_bytes(candidate.read_bytes())
            if digest != expected_record.get("sha256"):
                raise EpisodeContractError(
                    "AutoPET V minimal runtime file hash mismatch: "
                    + relative_candidate
                )
            file_records.append(
                {
                    "path": relative_candidate,
                    "sha256": digest,
                    "bytes": candidate.stat().st_size,
                }
            )
        if tuple(sorted(observed_files)) != AUTOPETV_RUNTIME_ALLOWLIST:
            raise EpisodeContractError("AutoPET V minimal runtime file inventory mismatch")
        if observed_dirs != {"interactive"}:
            raise EpisodeContractError("AutoPET V minimal runtime directory inventory mismatch")
        bundle_sha256 = _sha256_json(
            {
                "schema_version": AUTOPETV_RUNTIME_SCHEMA,
                "manifest_sha256": manifest_sha256,
                "files": sorted(file_records, key=lambda item: item["path"]),
            }
        )
        return {
            "repository": "lab-midas/autoPETV",
            "upstream_repository": AUTOPETV_RUNTIME_REPOSITORY,
            "commit": expected_commit,
            "relative_path": relative,
            "file_sha256": observed_sha256,
            "provenance_mode": "FROZEN_MINIMAL_RUNTIME",
            "git_worktree": "NOT_PRESENT_BY_MINIMAL_RUNTIME_POLICY",
            "license": AUTOPETV_RUNTIME_LICENSE,
            "runtime_manifest": {
                "path": str(manifest_path),
                "sha256": manifest_sha256,
                "schema_version": AUTOPETV_RUNTIME_SCHEMA,
                "status": AUTOPETV_RUNTIME_STATUS,
            },
            "runtime_bundle_sha256": bundle_sha256,
            "runtime_files": sorted(file_records, key=lambda item: item["path"]),
        }

    def git(*arguments: str) -> str:
        try:
            result = subprocess.run(
                ["git", "-C", str(repo_root), *arguments],
                check=True,
                capture_output=True,
                text=True,
                shell=False,
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            raise EpisodeContractError("cannot verify official simulator git provenance") from exc
        return result.stdout.strip()

    head = git("rev-parse", "HEAD")
    if head != expected_commit:
        raise EpisodeContractError("autoPET V repository HEAD differs from pinned commit")
    git("ls-files", "--error-unmatch", "--", relative)
    if git("status", "--porcelain", "--", relative):
        raise EpisodeContractError("official simulator tracked file is dirty")
    return {
        "repository": "lab-midas/autoPETV",
        "commit": head,
        "relative_path": relative,
        "file_sha256": observed_sha256,
        "provenance_mode": "CLEAN_GIT_WORKTREE",
        "git_worktree": "CLEAN_FOR_SIMULATOR_FILE",
    }


def load_official_simulator(
    path: Path,
    *,
    expected_commit: str = DEFAULT_UPSTREAM_COMMIT,
    expected_sha256: str = DEFAULT_SIMULATOR_SHA256,
    runtime_manifest: Path | None = None,
) -> Callable[..., Any]:
    """Load the pinned autoPET V residual simulator from an explicit file."""
    path = Path(path).resolve()
    provenance = official_simulator_provenance(
        path,
        expected_commit=expected_commit,
        expected_sha256=expected_sha256,
        runtime_manifest=runtime_manifest,
    )
    module_name = f"autopetv_simulator_{_sha256_bytes(str(path).encode('utf-8'))[:12]}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise EpisodeContractError(f"cannot load official simulator module: {path}")
    module = importlib.util.module_from_spec(spec)
    previous_dont_write_bytecode = sys.dont_write_bytecode
    try:
        # The server runtime is an exact three-file package.  Importing it must
        # not mutate that package by materializing ``__pycache__``.
        sys.dont_write_bytecode = True
        spec.loader.exec_module(module)
    except Exception as exc:
        raise EpisodeContractError(
            f"cannot import official simulator and its pinned dependencies: {exc}"
        ) from exc
    finally:
        sys.dont_write_bytecode = previous_dont_write_bytecode
    simulator = getattr(module, "simulate_scribble_from_label", None)
    if not callable(simulator):
        raise EpisodeContractError(
            "official module has no callable simulate_scribble_from_label"
        )
    setattr(simulator, "_petct_official_provenance", provenance)
    setattr(simulator, "_petct_official_module", module)
    return simulator


def _load_volume(
    path: Path,
) -> tuple[np.ndarray, np.ndarray | None, tuple[float, float] | None]:
    path = Path(path)
    if path.suffix.casefold() == ".npy":
        return np.load(path), None, None
    try:
        import nibabel as nib
    except ImportError as exc:
        raise EpisodeContractError("nibabel is required for NIfTI inputs") from exc
    image = nib.load(str(path))
    spacing = tuple(float(value) for value in image.header.get_zooms()[:2])
    return np.asanyarray(image.dataobj), np.asarray(image.affine), spacing


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--m0", type=Path, required=True)
    parser.add_argument("--experiment-config", type=Path, required=True)
    parser.add_argument("--official-simulator", type=Path, required=True)
    parser.add_argument(
        "--official-runtime-manifest",
        type=Path,
        default=DEFAULT_RUNTIME_MANIFEST,
    )
    parser.add_argument("--official-commit", default=DEFAULT_UPSTREAM_COMMIT)
    parser.add_argument("--episode-id", required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--patient-id", required=True)
    parser.add_argument("--operation", choices=("ADD", "REMOVE"), required=True)
    parser.add_argument("--lane", choices=("controlled", "natural"), required=True)
    parser.add_argument("--strategy", choices=(*STRATEGIES, "auto"), default="auto")
    parser.add_argument("--strategy-salt", default="PETCT-PILOT-v1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--montage-reference", required=True)
    parser.add_argument("--m0-provenance-json")
    parser.add_argument("--oof-ready", type=Path)
    parser.add_argument("--visible-root", type=Path, required=True)
    parser.add_argument("--eval-root", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        experiment_config_bytes = args.experiment_config.read_bytes()
        experiment_config = json.loads(experiment_config_bytes.decode("utf-8"))
        local_radius_mm = float(experiment_config["editor"]["local_radius_mm"])
        minimum_local_area_mm2 = float(
            experiment_config["editor"]["minimum_local_area_mm2"]
        )
        minimum_best_slice_pixels = validated_minimum_best_slice_pixels(
            experiment_config["scribble"]["minimum_best_slice_pixels"]
        )
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise EpisodeContractError("invalid frozen experiment config") from exc
    if (
        not np.isfinite(local_radius_mm)
        or local_radius_mm <= 0
        or not np.isfinite(minimum_local_area_mm2)
        or minimum_local_area_mm2 <= 0
    ):
        raise EpisodeContractError("frozen state-relative intent radii must be positive")
    m0_provenance = resolve_m0_provenance(
        lane=args.lane,
        provenance_json=args.m0_provenance_json,
        oof_ready=args.oof_ready,
        case_id=args.case_id,
        patient_id=args.patient_id,
        m0_path=args.m0,
    )
    if args.lane == "natural":
        if args.gt.is_symlink() or not args.gt.is_file():
            raise EpisodeContractError("natural GT must be a regular non-symlink file")
        if _sha256_bytes(args.gt.read_bytes()) != m0_provenance["input_gt_sha256"]:
            raise EpisodeContractError(
                "natural GT hash differs from OOF input_gt_sha256 provenance"
            )
    gt, gt_affine, gt_spacing_xy = _load_volume(args.gt)
    m0, m0_affine, m0_spacing_xy = _load_volume(args.m0)
    if gt_spacing_xy is None or m0_spacing_xy is None:
        raise EpisodeContractError(
            "formal episode CLI requires NIfTI GT/M0 with physical spacing; "
            "NumPy arrays are fixture-only inputs for library tests"
        )
    if (
        gt_affine is not None
        and m0_affine is not None
        and not np.allclose(gt_affine, m0_affine, atol=1e-3, rtol=0)
    ):
        raise EpisodeContractError("GT/M0 affine mismatch")
    if not np.allclose(gt_spacing_xy, m0_spacing_xy, atol=1e-6, rtol=0):
        raise EpisodeContractError("GT/M0 in-plane spacing mismatch")
    residual = (
        compute_fn_residual(gt, m0)
        if args.operation == "ADD"
        else compute_fp_residual(gt, m0)
    )
    strategy = (
        assign_scribble_strategy(args.patient_id, salt=args.strategy_salt)
        if args.strategy == "auto"
        else args.strategy
    )
    simulator = load_official_simulator(
        args.official_simulator,
        expected_commit=args.official_commit,
        runtime_manifest=args.official_runtime_manifest,
    )
    scribble = generate_residual_scribble(
        residual,
        operation=args.operation,
        strategy=strategy,
        seed=args.seed,
        simulator=simulator,
        upstream_commit=args.official_commit,
        minimum_best_slice_pixels=minimum_best_slice_pixels,
    )
    scribble["official_source_provenance"] = getattr(
        simulator, "_petct_official_provenance"
    )
    # Import lazily to avoid the dataset module's intentional reuse of this
    # module.  The formal path never accepts a caller-provided goal: target
    # and scope are derived from the actual GT/M0 state and generated scribble.
    from data.build_petct_scribble_dataset import derive_goal_and_authorized_target

    try:
        derived_goal, authorized_target, target_stats = (
            derive_goal_and_authorized_target(
                gt=gt,
                m0=m0,
                operation=args.operation,
                coordinates_xyz=scribble["coordinates_xyz"],
                spacing_xy=gt_spacing_xy,
                local_radius_mm=local_radius_mm,
                minimum_local_area_mm2=minimum_local_area_mm2,
            )
        )
    except RuntimeError as exc:
        raise EpisodeContractError(
            f"state-relative intent derivation failed: {exc}"
        ) from exc
    patient_group_hash = _sha256_bytes(
        f"PETCT-PATIENT-GROUP-v1|{args.patient_id}".encode("utf-8")
    )
    visible, evaluation = build_episode_documents(
        episode_id=args.episode_id,
        lane=args.lane,
        patient_group_hash=patient_group_hash,
        montage_reference=args.montage_reference,
        m0_provenance=m0_provenance,
        scribble_record=scribble,
        source_case_id=args.case_id,
        source_patient_id=args.patient_id,
        residual_sha256=scribble["residual_sha256"],
        residual_voxels=scribble["residual_voxels"],
        gold_intent=canonical_intent_frame(derived_goal),
    )
    evaluation["state_relative_derivation"] = {
        "contract": "DERIVED_FROM_ACTUAL_GT_M0_AND_GENERATED_SCRIBBLE",
        "goal": derived_goal,
        "authorized_target_sha256": _mask_sha256(authorized_target),
        "authorized_target_voxels": int(np.asarray(authorized_target).sum()),
        "spacing_xy_mm": list(gt_spacing_xy),
        "local_radius_mm": local_radius_mm,
        "minimum_local_area_mm2": minimum_local_area_mm2,
        "experiment_config_sha256": _sha256_bytes(experiment_config_bytes),
        "target_stats": target_stats,
    }
    receipt = publish_episode_documents(
        visible,
        evaluation,
        visible_root=args.visible_root,
        eval_root=args.eval_root,
    )
    sys.stdout.write(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
