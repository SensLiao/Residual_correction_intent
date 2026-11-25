"""The simulated user: one scribble per round, from the official autoPET V scribble simulator.

Every scribble is drawn by the official simulator (``sirb.third_party.autopet_v.simulate_scribbles``, unmodified)
with seed :data:`SIMULATOR_SEED` in one of its three drawing styles (:data:`STYLES`). Given a binary error region, the
simulator finds the largest 2D connected component (8-connected) over all axial slices and draws a scribble in it;
it also returns the scribble's size.

:func:`choose_correction` is the official decision of a round: the simulator is shown the false positives (mask minus
ground truth) and the false negatives (ground truth minus mask) separately, and the scribble on the false negatives
is chosen, as an addition scribble, when the false-positive candidate is not longer than the false-negative one;
otherwise the scribble on the false positives is chosen, as a removal scribble. :class:`ScribbleSimulator` draws:

* :meth:`~ScribbleSimulator.draw`, the simulated user of the evaluation: the official decision on the current mask.
  When the mask has no error left there is no valid scribble (``valid`` is false, the correction type ``None``);
* :meth:`~ScribbleSimulator.draw_correction_type`: the same scribble with the correction type imposed, drawn on the
  errors of that correction type only (a type without error gives no valid scribble);
* :meth:`~ScribbleSimulator.draw_on_error`: a scribble of a given correction type on exactly the given error voxels.

Arrays are ZYX on the native PET grid. The official simulator takes XYZ arrays and slices along its third axis, which
is Z here, so every scribble lies on one axial slice. The official code reseeds Python's ``random`` module on every
random or boundary scribble; the global Python and NumPy random states are restored after each call, so a caller is
never reseeded. The simulator may read the ground truth; it returns a scribble and its correction type only.
"""

from __future__ import annotations

import contextlib
import random
from dataclasses import dataclass
from typing import Any, Iterator

import numpy as np

from ..data.roles import correction_sign

__all__ = ["ADDITION", "REMOVAL", "SIMULATOR_SEED", "STYLES", "NoScribbleDrawn", "Scribble", "ScribbleSimulator",
           "SimulatorContractError", "choose_correction"]

#: The drawing styles of the official simulator.
STYLES = ("centerline", "random", "boundary")
#: The seed passed to the official simulator on every call.
SIMULATOR_SEED = 42
#: Correction types: an addition scribble (+1, drawn on false negatives) and a removal scribble (-1, false positives).
ADDITION, REMOVAL = 1, -1


class SimulatorContractError(RuntimeError):
    """The official simulator returned something its contract does not allow."""


class NoScribbleDrawn(SimulatorContractError):
    """The official simulator drew no voxel although the error region it was shown is not empty."""


@dataclass(frozen=True)
class Scribble:
    """One scribble of the simulated user and how it was chosen.

    ``coordinates_zyx`` lists the scribble voxels in ``np.argwhere`` order; ``scribble_zyx`` is the same set as a
    volume. The two candidate sizes are those of the official decision that produced the scribble (a side that was not
    shown to the simulator reports 0). A scribble is valid when it has at least one voxel.
    """

    correction_type: int | None
    scribble_zyx: np.ndarray
    coordinates_zyx: np.ndarray
    style: str
    false_positive_candidate_size: int
    false_negative_candidate_size: int

    @property
    def valid(self) -> bool:
        return len(self.coordinates_zyx) > 0


def _binary(array: Any, name: str) -> np.ndarray:
    value = np.asarray(array)
    if value.ndim != 3:
        raise ValueError(f"{name} must be a 3D array, got {value.ndim} dimensions")
    if value.dtype != np.bool_:
        if not ((value == 0) | (value == 1)).all():
            raise ValueError(f"{name} must hold only 0 and 1")
        value = value.astype(bool)
    return value


def _pair(first: Any, second: Any, names: tuple[str, str]) -> tuple[np.ndarray, np.ndarray]:
    a, b = _binary(first, names[0]), _binary(second, names[1])
    if a.shape != b.shape:
        raise ValueError(f"{names[0]} shape {a.shape} differs from {names[1]} shape {b.shape}")
    return a, b


def _check_style(style: str) -> str:
    # The official simulator treats an unknown style as "random"; refuse it here instead.
    if style not in STYLES:
        raise ValueError(f"drawing style must be one of {STYLES}, got {style!r}")
    return style


def _official_simulator() -> Any:
    """The official simulator module; it imports nibabel, networkx, cc3d and scikit-image, so it is loaded on the
    first scribble rather than when this module is imported."""
    from ..third_party.autopet_v import simulate_scribbles  # noqa: PLC0415

    return simulate_scribbles


@contextlib.contextmanager
def _restored_random_states() -> Iterator[None]:
    python_state, numpy_state = random.getstate(), np.random.get_state()
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def _candidate(error_xyz: np.ndarray, style: str, seed: int) -> tuple[list[list[int]], int]:
    """(XYZ coordinates, size) of the official scribble on one error region; ([], 0) for an empty region."""
    if not np.any(error_xyz):
        return [], 0
    result = _official_simulator().simulate_scribble_from_label(error_xyz.astype(np.uint8), style, int(seed))
    if not isinstance(result, tuple) or len(result) not in (2, 3):
        raise SimulatorContractError(f"the official simulator returned an unknown value: {type(result).__name__}")
    coordinates = [[int(value) for value in coordinate] for coordinate in result[0]]
    size = 0 if len(result) == 2 else int(result[2])
    if any(len(coordinate) != 3 for coordinate in coordinates):
        raise SimulatorContractError("the official simulator returned a coordinate that is not 3D")
    if size != len(coordinates):
        raise SimulatorContractError(f"the official simulator returned {len(coordinates)} coordinates for a "
                                     f"scribble of size {size}")
    return coordinates, size


def _coordinates_zyx(coordinates_xyz: list[list[int]], shape: tuple[int, ...]) -> np.ndarray:
    """Official (x, y, z) coordinates as unique ZYX rows in ``np.argwhere`` order."""
    zyx = np.ascontiguousarray(np.asarray(coordinates_xyz, dtype=np.int64).reshape(-1, 3)[:, ::-1])
    if len(zyx):
        outside = np.any((zyx < 0) | (zyx >= np.asarray(shape, dtype=np.int64)), axis=1)
        if outside.any():
            raise SimulatorContractError(f"the official simulator drew outside the volume: first ZYX voxel "
                                         f"{zyx[outside][0].tolist()}, volume shape {tuple(shape)}")
        zyx = np.unique(zyx, axis=0)
    return zyx


def _volume(coordinates_zyx: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    volume = np.zeros(shape, dtype=bool)
    if len(coordinates_zyx):
        volume[tuple(coordinates_zyx.T)] = True
    return volume


def choose_correction(prediction_zyx: Any, ground_truth_zyx: Any, *, style: str,
                      seed: int = SIMULATOR_SEED) -> Scribble:
    """The official decision of one round on ``prediction_zyx`` (module docstring).

    The correction type is ``None`` only when both error sides are empty; the returned scribble may still be empty
    when the simulator drew nothing on the chosen side (callers decide whether that is allowed).
    """
    prediction, truth = _pair(prediction_zyx, ground_truth_zyx, ("prediction", "ground truth"))
    _check_style(style)
    # The official code receives XYZ uint8 C-contiguous arrays, as the official evaluation passes them.
    prediction_xyz = np.ascontiguousarray(np.transpose(prediction.astype(np.uint8), (2, 1, 0))) > 0
    truth_xyz = np.ascontiguousarray(np.transpose(truth.astype(np.uint8), (2, 1, 0))) > 0
    with _restored_random_states():
        removal, fp_size = _candidate(prediction_xyz & ~truth_xyz, style, seed)
        addition, fn_size = _candidate(~prediction_xyz & truth_xyz, style, seed)
    if fp_size <= fn_size:
        correction_type, coordinates = ADDITION, addition
    else:
        correction_type, coordinates = REMOVAL, removal
    if not (prediction ^ truth).any():
        correction_type = None
    zyx = _coordinates_zyx(coordinates, prediction.shape)
    return Scribble(correction_type, _volume(zyx, prediction.shape), zyx, style, fp_size, fn_size)


class ScribbleSimulator:
    """The simulated user (module docstring); ``seed`` is passed to the official simulator on every call."""

    def __init__(self, seed: int = SIMULATOR_SEED) -> None:
        if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
            raise TypeError(f"seed must be an integer, got {seed!r}")
        self.seed = int(seed)

    def draw(self, mask_zyx: Any, ground_truth_zyx: Any, style: str) -> Scribble:
        """The scribble of the official decision on the current mask; no valid scribble when no error is left."""
        mask, truth = _pair(mask_zyx, ground_truth_zyx, ("mask", "ground truth"))
        scribble = choose_correction(mask, truth, style=style, seed=self.seed)
        if scribble.correction_type is None:
            return scribble
        if not scribble.valid:
            raise NoScribbleDrawn(f"the official simulator drew nothing although {int((mask & ~truth).sum())} "
                                  f"false-positive and {int((truth & ~mask).sum())} false-negative voxels remain "
                                  f"(candidate sizes {scribble.false_positive_candidate_size} and "
                                  f"{scribble.false_negative_candidate_size})")
        return scribble

    def draw_correction_type(self, mask_zyx: Any, ground_truth_zyx: Any, correction_type: Any, style: str) -> Scribble:
        """The scribble the official decision would draw for ``correction_type``, drawn on the errors of that type
        only: the false negatives for an addition, the false positives for a removal."""
        mask, truth = _pair(mask_zyx, ground_truth_zyx, ("mask", "ground truth"))
        _check_style(style)
        error = truth & ~mask if correction_sign(correction_type) == ADDITION else mask & ~truth
        return self.draw_on_error(error, truth, correction_type, style)

    def draw_on_error(self, error_zyx: Any, ground_truth_zyx: Any, correction_type: Any, style: str) -> Scribble:
        """A scribble of ``correction_type`` on exactly ``error_zyx``, an error of that type for the ground truth
        (inside it for an addition, outside it for a removal).

        The simulator is shown a mask whose only error of that type is ``error_zyx`` and which has no error of the
        other type: the ground truth without the error (addition) or with it (removal). An empty error gives no valid
        scribble with the correction type kept.
        """
        error, truth = _pair(error_zyx, ground_truth_zyx, ("error", "ground truth"))
        _check_style(style)
        sign = correction_sign(correction_type)
        stray = error & ~truth if sign == ADDITION else error & truth
        if stray.any():
            raise ValueError(f"the error is not an {'addition' if sign == ADDITION else 'removal'} error of the ground "
                             f"truth: {int(stray.sum())} of its {int(error.sum())} voxels are "
                             f"{'outside' if sign == ADDITION else 'inside'} the ground truth")
        shown = truth & ~error if sign == ADDITION else truth | error
        scribble = choose_correction(shown, truth, style=style, seed=self.seed)
        if not scribble.valid:
            if error.any():
                raise NoScribbleDrawn(f"the official simulator drew nothing on an error of {int(error.sum())} voxels "
                                      f"(candidate sizes {scribble.false_positive_candidate_size} and "
                                      f"{scribble.false_negative_candidate_size})")
            return Scribble(sign, scribble.scribble_zyx, scribble.coordinates_zyx, style,
                            scribble.false_positive_candidate_size, scribble.false_negative_candidate_size)
        if scribble.correction_type != sign:
            raise SimulatorContractError(f"the imposed correction type {sign:+d} was not drawn: the official decision "
                                         f"chose {scribble.correction_type:+d}")
        return scribble
