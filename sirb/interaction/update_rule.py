"""The update rule: how SIRB's target probability becomes the updated mask, and the interaction history.

A scribble may change only its editable domain: the voxels outside the current mask for an addition scribble, the
voxels inside it for a removal scribble (and, when the examination marks valid native voxels, only valid ones). The
update rule applies every editable voxel whose target probability is at least :data:`TARGET_THRESHOLD` (0.5) with
the scribble's correction type: an addition adds those voxels to the mask and never removes one; a removal removes
them and never adds one. The scribble itself is not forced into the correction, earlier scribbles are not protected,
and no ground truth is read.

The interaction history SIRB reads in the next rounds is built here too: the earlier scribbles of each correction
type, in which the latest scribble owns a voxel that two scribbles of different correction types marked
(:func:`accumulate_scribbles`; the current scribble is never part of its own history).

Every function works on NumPy arrays and, except :func:`update_mask` and :func:`accumulate_scribbles`, also on
boolean torch tensors (the predictor applies the rule on its device).
"""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from ..data.roles import correction_sign

__all__ = ["ADDITION", "REMOVAL", "TARGET_THRESHOLD", "accept", "accumulate_scribbles", "apply", "editable_domain",
           "update_mask"]

#: Correction types: an addition scribble (+1) and a removal scribble (-1).
ADDITION, REMOVAL = 1, -1
#: An editable voxel is applied when its target probability is at least this value.
TARGET_THRESHOLD = 0.5


def _copy(array: Any) -> Any:
    return array.copy() if isinstance(array, np.ndarray) else array.clone()


def editable_domain(mask: Any, correction_type: Any, valid: Any = None) -> Any:
    """The voxels a scribble of ``correction_type`` may change: ``~mask`` (addition) or ``mask`` (removal), and
    ``valid`` when given. Returns a new array (or tensor)."""
    domain = ~mask if correction_sign(correction_type) == ADDITION else _copy(mask)
    return domain if valid is None else domain & valid


def apply(mask: Any, region: Any, correction_type: Any) -> Any:
    """The mask after applying ``region`` with ``correction_type``: ``mask | region`` (addition) or
    ``mask & ~region`` (removal)."""
    return mask | region if correction_sign(correction_type) == ADDITION else mask & ~region


def accept(target_probability: Any) -> Any:
    """Whether each target probability reaches the update rule's threshold (``>= TARGET_THRESHOLD``)."""
    return target_probability >= TARGET_THRESHOLD


def _binary(array: Any, name: str) -> np.ndarray:
    value = np.asarray(array)
    if value.ndim != 3:
        raise ValueError(f"{name} must be a 3D array, got {value.ndim} dimensions")
    if value.dtype != np.bool_:
        if not ((value == 0) | (value == 1)).all():
            raise ValueError(f"{name} must hold only 0 and 1")
        value = value.astype(bool)
    return value


def update_mask(mask_zyx: Any, target_probability_zyx: Any, correction_type: Any, *,
                valid_zyx: Any = None) -> tuple[np.ndarray, np.ndarray]:
    """(updated mask, applied correction) of the update rule on a target-probability volume.

    The applied correction is the editable voxels whose target probability is at least TARGET_THRESHOLD; the
    target probability must be finite and lie in [0, 1] wherever it is editable.
    """
    mask = _binary(mask_zyx, "mask")
    probability = np.asarray(target_probability_zyx)
    if probability.shape != mask.shape:
        raise ValueError(f"target probability shape {probability.shape} differs from the mask shape {mask.shape}")
    valid = None if valid_zyx is None else _binary(valid_zyx, "valid")
    if valid is not None and valid.shape != mask.shape:
        raise ValueError(f"valid shape {valid.shape} differs from the mask shape {mask.shape}")
    domain = editable_domain(mask, correction_type, valid)
    editable = probability[domain]
    if not np.isfinite(editable).all() or np.any((editable < 0) | (editable > 1)):
        raise ValueError("target probabilities must be finite and lie in [0, 1] in the editable domain")
    applied = domain & accept(probability)
    return apply(mask, applied, correction_type), applied


def accumulate_scribbles(earlier: Iterable[tuple[Any, Any]], *, shape) -> tuple[np.ndarray, np.ndarray]:
    """(earlier addition scribbles, earlier removal scribbles) from the earlier scribbles of an examination.

    ``earlier`` lists ``(correction_type, scribble_zyx)`` in round order, only scribbles that were applied, never the
    current one. A voxel marked by scribbles of both correction types belongs to the latest of them.
    """
    grid = tuple(int(v) for v in shape)
    addition, removal = np.zeros(grid, dtype=bool), np.zeros(grid, dtype=bool)
    for correction_type, scribble in earlier:
        marks = _binary(scribble, "earlier scribble")
        if marks.shape != grid:
            raise ValueError(f"earlier scribble shape {marks.shape} differs from {grid}")
        own, other = (addition, removal) if correction_sign(correction_type) == ADDITION else (removal, addition)
        own |= marks
        other &= ~marks
    return addition, removal
