"""Correction sessions: one examination corrected over rounds 0 to N.

Round 0 is the initial segmentation. In each round t = 1..N the simulated user draws one scribble on the current mask
(``sirb.interaction.scribble_simulator``, in the examination's drawing style); SIRB predicts the target probability
and the update rule applies the correction; the updated mask is the current mask of round t + 1. A round without a
valid scribble (no error left; or, with ``draw_on_empty_ground_truth`` false, an examination whose ground truth is
empty, where the simulated user is not asked at all) keeps the current mask, does not call the model and is recorded
as skipped.

What SIRB reads besides the current mask and scribble is the interaction history (:class:`InteractionHistory`): the
earlier valid scribbles of each correction type (``update_rule.accumulate_scribbles``) and the previous mask, the
mask before the last correction that was applied; it is unknown in round 1 and after a skipped round. The ground
truth reaches only the simulated user, never the predictor.

:func:`run_session` drives any predictor with a ``predict_state(state) -> Prediction`` method
(``sirb.sirb_predictor.SIRBPredictor``) and returns the masks after rounds 0 to N, which the evaluation scores, with
the scribbles, correction types and skipped rounds.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from ..data.geometry import validate_affine_xyz
from ..data.observations import State
from ..data.roles import correction_sign
from .update_rule import accumulate_scribbles

__all__ = ["Examination", "InteractionHistory", "Round", "SessionResult", "run_session"]

#: Why a round was skipped.
NO_ERROR_LEFT = "no error left"
EMPTY_GROUND_TRUTH = "empty ground truth"


@dataclass(frozen=True)
class Examination:
    """One examination as SIRB reads it: normalised PET and CT on the native PET grid (ZYX), the XYZ voxel-to-world
    affine, the initial segmentation (round 0), the valid native voxels (all when ``None``) and the PET pad value.

    The arrays are kept as given (no copy); ``valid_zyx=None`` becomes an all-true array once, so that every state of
    the examination shares the same arrays. No ground truth belongs here.
    """

    pet_zyx: np.ndarray
    ct_zyx: np.ndarray
    affine_xyz: np.ndarray
    initial_segmentation_zyx: np.ndarray
    valid_zyx: np.ndarray | None = None
    pet_pad_value: float = 0.0
    examination_id: str = ""
    patient_id: str = ""

    def __post_init__(self) -> None:
        pet, ct = np.asarray(self.pet_zyx), np.asarray(self.ct_zyx)
        if pet.ndim != 3 or ct.shape != pet.shape:
            raise ValueError(f"PET and CT must be matching 3D ZYX volumes; got {pet.shape} and {ct.shape}")
        mask = np.asarray(self.initial_segmentation_zyx, dtype=bool)
        valid = np.ones(pet.shape, dtype=bool) if self.valid_zyx is None else np.asarray(self.valid_zyx, dtype=bool)
        for name, value in (("initial segmentation", mask), ("valid", valid)):
            if value.shape != pet.shape:
                raise ValueError(f"{name} shape {value.shape} differs from the PET shape {pet.shape}")
        object.__setattr__(self, "pet_zyx", pet)
        object.__setattr__(self, "ct_zyx", ct)
        object.__setattr__(self, "affine_xyz", validate_affine_xyz(self.affine_xyz))
        object.__setattr__(self, "initial_segmentation_zyx", mask)
        object.__setattr__(self, "valid_zyx", valid)
        object.__setattr__(self, "pet_pad_value", float(self.pet_pad_value))

    @classmethod
    def from_inputs(cls, inputs: Any) -> "Examination":
        """The examination of one cached examination (``sirb.data.preprocessing.ExaminationCache.inputs``)."""
        return cls(inputs.pet, inputs.ct, inputs.affine_xyz, inputs.initial_segmentation,
                   pet_pad_value=inputs.pet_pad_value, examination_id=inputs.examination_id,
                   patient_id=inputs.patient_id)

    @property
    def shape_zyx(self) -> tuple[int, int, int]:
        return tuple(int(v) for v in self.pet_zyx.shape)


class InteractionHistory:
    """The current mask of an examination and what SIRB remembers of its earlier rounds (module docstring)."""

    def __init__(self, initial_mask_zyx: Any) -> None:
        self.mask = np.array(initial_mask_zyx, dtype=bool, copy=True)
        self.previous_mask: np.ndarray | None = None
        self.scribbles: list[tuple[int, np.ndarray]] = []

    def state(self, examination: Examination, scribble_zyx: Any, correction_type: Any) -> State:
        """The state of a round in which ``scribble_zyx`` of ``correction_type`` is drawn on the current mask."""
        addition, removal = accumulate_scribbles(self.scribbles, shape=self.mask.shape)
        return State(examination.pet_zyx, examination.ct_zyx, self.mask, scribble_zyx, examination.affine_xyz,
                     correction_type, addition, removal, self.previous_mask, examination.valid_zyx,
                     examination.examination_id, examination.patient_id, examination.pet_pad_value)

    def record(self, scribble_zyx: Any, correction_type: Any, updated_mask_zyx: Any) -> None:
        """A round whose scribble was applied: the updated mask becomes the current mask."""
        updated = np.array(updated_mask_zyx, dtype=bool, copy=True)
        if updated.shape != self.mask.shape:
            raise ValueError(f"updated mask shape {updated.shape} differs from {self.mask.shape}")
        self.previous_mask = self.mask
        self.mask = updated
        self.scribbles.append((correction_sign(correction_type), np.array(scribble_zyx, dtype=bool, copy=True)))

    def skip(self) -> None:
        """A round without a valid scribble: the mask stays and the previous mask becomes unknown."""
        self.previous_mask = None


@dataclass(frozen=True)
class Round:
    """One round t = 1..N: whether it was skipped (and why), the scribble's correction type and voxel count, the
    voxels the correction changed, its wall-clock seconds and the predictor's meta information."""

    index: int
    skipped: bool
    correction_type: int | None
    scribble_voxels: int
    applied_voxels: int
    seconds: float
    reason: str = ""
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"round": self.index, "skipped": self.skipped, "correction_type": self.correction_type,
                "scribble_voxels": self.scribble_voxels, "applied_voxels": self.applied_voxels,
                "seconds": self.seconds, "reason": self.reason, "meta": dict(self.meta)}


@dataclass(frozen=True)
class SessionResult:
    """The masks after rounds 0 to N (``masks[0]`` is the initial segmentation), the scribble of each round t = 1..N
    (``None`` when skipped) and the round records."""

    examination_id: str
    patient_id: str
    style: str
    masks: tuple[np.ndarray, ...]
    scribbles: tuple[np.ndarray | None, ...]
    rounds: tuple[Round, ...]

    @property
    def skipped(self) -> list[bool]:
        return [record.skipped for record in self.rounds]

    @property
    def correction_types(self) -> list[int | None]:
        return [record.correction_type for record in self.rounds]


class Predictor(Protocol):
    def predict_state(self, state: State) -> Any: ...


def run_session(predictor: Predictor, examination: Examination, ground_truth_zyx: Any, *, simulator: Any, style: str,
                rounds: int, draw_on_empty_ground_truth: bool) -> SessionResult:
    """Correct ``examination`` for ``rounds`` rounds with the simulated user ``simulator`` (module docstring).

    ``draw_on_empty_ground_truth`` decides what happens on an examination without a lesion: false skips every round
    (the scribble simulator is not called); true lets the simulated user draw removal scribbles on its false
    positives like on any other examination.
    """
    if isinstance(rounds, bool) or not isinstance(rounds, int) or rounds < 1:
        raise ValueError(f"rounds must be a positive integer, got {rounds!r}")
    if not isinstance(draw_on_empty_ground_truth, bool):
        raise ValueError(f"draw_on_empty_ground_truth must be true or false, got {draw_on_empty_ground_truth!r}")
    truth = np.asarray(ground_truth_zyx, dtype=bool)
    if truth.shape != examination.shape_zyx:
        raise ValueError(f"ground truth shape {truth.shape} differs from the examination shape "
                         f"{examination.shape_zyx}")
    history = InteractionHistory(examination.initial_segmentation_zyx)
    masks, scribbles, records = [history.mask.copy()], [], []
    skip_all = not truth.any() and not draw_on_empty_ground_truth
    for index in range(1, rounds + 1):
        started = time.perf_counter()
        scribble = None if skip_all else simulator.draw(history.mask, truth, style)
        if scribble is None or not scribble.valid:
            history.skip()
            masks.append(history.mask.copy())
            scribbles.append(None)
            records.append(Round(index, True, None, 0, 0, time.perf_counter() - started,
                                 EMPTY_GROUND_TRUTH if skip_all else NO_ERROR_LEFT))
            continue
        state = history.state(examination, scribble.scribble_zyx, scribble.correction_type)
        prediction = predictor.predict_state(state)
        applied = int((prediction.updated_mask ^ history.mask).sum())
        history.record(scribble.scribble_zyx, scribble.correction_type, prediction.updated_mask)
        masks.append(history.mask.copy())
        scribbles.append(scribble.scribble_zyx.copy())
        records.append(Round(index, False, int(scribble.correction_type), int(scribble.scribble_zyx.sum()), applied,
                             time.perf_counter() - started, "", dict(prediction.meta)))
    return SessionResult(examination.examination_id, examination.patient_id, style, tuple(masks), tuple(scribbles),
                         tuple(records))
