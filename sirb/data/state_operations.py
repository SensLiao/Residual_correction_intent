"""Volume operations and configuration checks shared by training state generators."""

from typing import Any, Mapping

import numpy as np
from scipy import ndimage

CONNECTIVITY_RANK = {6: 1, 18: 2, 26: 3}


class StateError(RuntimeError):
    """A state, state pair or stored record is invalid for the requested operation."""


def binary_volume(array, name: str) -> np.ndarray:
    """Validate a binary volume and return a boolean array on the same grid."""
    a = np.asarray(array)
    if a.ndim != 3:
        raise StateError(f"{name} must be a 3D array, got ndim={a.ndim}")
    if a.dtype != np.bool_ and not np.all((a == 0) | (a == 1)):
        raise StateError(f"{name} must be binary")
    return a.astype(bool, copy=False)


def require_same_shape(*arrays) -> None:
    """Require a shared shape across arrays on the same examination grid."""
    shapes = {np.shape(a) for a in arrays}
    if len(shapes) != 1:
        raise StateError(f"arrays must share one shape, got {sorted(shapes)}")


def readonly_volume(volume: np.ndarray) -> np.ndarray:
    """Mark a volume as read-only before it enters a record: records share volumes."""
    volume.setflags(write=False)
    return volume


def connectivity_structure(connectivity: int) -> np.ndarray:
    """Return the 3D neighbourhood for 6, 18 or 26 voxel connectivity."""
    return ndimage.generate_binary_structure(3, CONNECTIVITY_RANK[connectivity])


def require_generator(rng) -> np.random.Generator:
    """Validate and return the NumPy generator used by the sampling operation."""
    if not isinstance(rng, np.random.Generator):
        raise StateError(f"rng must be a numpy.random.Generator, got {type(rng).__name__}")
    return rng


def bounding_box(region, margin=0) -> tuple[slice, ...]:
    """Smallest box holding every True voxel of a non-empty region, grown by ``margin`` and clipped to the array."""
    occupied = np.asarray(region, dtype=bool)
    grow = np.broadcast_to(np.asarray(margin), (occupied.ndim,))
    box = []
    for axis in range(occupied.ndim):
        present = np.flatnonzero(occupied.any(axis=tuple(a for a in range(occupied.ndim) if a != axis)))
        box.append(slice(max(int(present[0]) - int(grow[axis]), 0),
                         min(int(present[-1]) + int(grow[axis]) + 1, occupied.shape[axis])))
    return tuple(box)


def embed_region(shape, box, part) -> np.ndarray:
    """Place a cropped boolean region in a volume of the given shape."""
    volume = np.zeros(tuple(shape), dtype=bool)
    volume[box] = part
    return volume


def configured_positive_integer(config: Mapping[str, Any], key: str) -> int:
    """Read a positive integer from the configuration."""
    value = config[key]
    if isinstance(value, bool) or int(value) != value or int(value) < 1:
        raise StateError(f"{key} must be a positive integer, got {value!r}")
    return int(value)


def configured_probability(config: Mapping[str, Any], key: str) -> float:
    """Read a probability in the closed interval from zero to one."""
    value = float(config[key])
    if not 0.0 <= value <= 1.0:
        raise StateError(f"{key} must lie in [0, 1], got {value}")
    return value


def configured_fraction_range(config: Mapping[str, Any], key: str) -> tuple[float, float]:
    """Read an ordered fractional interval with positive endpoints."""
    low, high = (float(v) for v in config[key])
    if not 0.0 < low <= high <= 1.0:
        raise StateError(f"{key} must satisfy 0 < low <= high <= 1, got {(low, high)}")
    return low, high
