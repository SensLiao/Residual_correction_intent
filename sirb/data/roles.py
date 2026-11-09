"""The three roles of a scribble: target error, other errors and correct tissue.

For one current mask, ground truth and scribble with its correction type, every editable voxel (outside the mask
for an addition scribble, inside it for a removal scribble) gets one role on the native PET grid:

* target error: the error components of the scribble's correction type that the scribble overlaps, where an error
  component is an 18-connected region of the editable domain that disagrees with the ground truth;
* other errors: every other error voxel of that correction type;
* correct tissue: the editable voxels the mask already labels correctly.

Voxels outside the editable domain are ``IGNORE``. Roles are training labels; they are never a model input.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from .geometry import CONNECTIVITY_18

__all__ = ["CORRECT_TISSUE", "IGNORE", "OTHER_ERRORS", "ROLE_COUNT", "ROLE_NAMES", "TARGET_ERROR", "Roles",
           "correction_sign", "make_roles"]

#: Role indices; role logits and probabilities use this channel order.
TARGET_ERROR, OTHER_ERRORS, CORRECT_TISSUE, IGNORE = 0, 1, 2, -1
ROLE_COUNT = 3
ROLE_NAMES = ("target error", "other errors", "correct tissue")

_ADDITION = (1, "+", "ADD", "add", "addition")
_REMOVAL = (-1, "-", "REMOVE", "remove", "removal")


def correction_sign(correction_type) -> int:
    """+1 for an addition scribble, -1 for a removal scribble (accepts +1/-1, "addition"/"removal", "ADD"/"REMOVE")."""
    if correction_type in _ADDITION:
        return 1
    if correction_type in _REMOVAL:
        return -1
    raise ValueError(f"correction type actual={correction_type!r}, expected addition (+1) or removal (-1)")


@dataclass(frozen=True)
class Roles:
    """Roles of one state and scribble on the native grid (ZYX arrays)."""
    labels: np.ndarray             # int64 role index per voxel, IGNORE outside the editable domain
    target_error: np.ndarray
    other_errors: np.ndarray
    correct_tissue: np.ndarray
    editable: np.ndarray
    errors: np.ndarray
    components: np.ndarray         # 18-connected labels of ``errors``
    touched_components: tuple[int, ...]


def make_roles(mask_zyx, ground_truth_zyx, scribble_zyx, correction_type, *, valid_zyx=None) -> Roles:
    """Roles of every voxel for a scribble of ``correction_type`` drawn on ``mask_zyx`` (module docstring)."""
    mask, truth, scribble = (np.asarray(v, dtype=bool) for v in (mask_zyx, ground_truth_zyx, scribble_zyx))
    if mask.ndim != 3 or truth.shape != mask.shape or scribble.shape != mask.shape:
        raise ValueError("roles require matching full-volume ZYX mask, ground truth and scribble")
    valid = np.ones_like(mask) if valid_zyx is None else np.asarray(valid_zyx, dtype=bool)
    if valid.shape != mask.shape:
        raise ValueError("roles valid mask shape differs from the native state")
    addition = correction_sign(correction_type) == 1
    editable = (~mask if addition else mask) & valid
    errors = editable & (truth if addition else ~truth)
    components, _ = ndimage.label(errors, structure=CONNECTIVITY_18)
    touched = np.unique(components[scribble & valid])
    touched = touched[touched != 0]
    target = np.isin(components, touched) & errors
    other = errors & ~target
    correct = editable & ~errors
    labels = np.full(mask.shape, IGNORE, dtype=np.int64)
    labels[target], labels[other], labels[correct] = TARGET_ERROR, OTHER_ERRORS, CORRECT_TISSUE
    return Roles(labels, target, other, correct, editable, errors, components, tuple(int(v) for v in touched))
