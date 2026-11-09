"""Evaluation metrics of the masks of one examination: Dice, DMM, false-positive components and volume, and nAUC.

Each mask is scored on the native PET grid with the official autoPET V metric module,
``sirb.third_party.autopet_v.metrics`` (unmodified):

* Dice is its global Dice, 2 TP / (2 TP + FP + FN) over the voxels.
* DMM, its detection-matching metric, is a lesion-level F1 score. Lesions are the 18-connected components of the
  ground truth and predicted components those of the mask (voxels that share a face or an edge are connected). A
  predicted component matches a lesion when their intersection over union is at least 0.1, and DMM takes the Dice form
  with the counts of matched lesions, unmatched predicted components and unmatched lesions.
* False-positive components are the unmatched predicted components, the false positives of DMM.
* False-positive volume is the volume of all predicted voxels outside the ground truth, in mL. It is not the
  whole-component false-positive volume of the official module.
* nAUC is the normalised area under the curve of Dice or DMM over rounds 0 to N, (1/N) sum_t (q[t-1] + q[t]) / 2.

An examination with empty ground truth has no Dice, DMM or nAUC (``None``, the official module's NaN); its
false-positive components and volume are defined. Masks are 3D arrays of booleans, or of 0 and 1, on one grid.

``score_examination`` returns the scores of one examination as a JSON-ready dictionary with the keys below; the
summaries over examinations and patients are in ``sirb.evaluation.protocol``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

__all__ = ["CONNECTIVITY", "DICE", "DMM", "DMM_IOU_THRESHOLD", "EXAMINATION_METRICS", "FALSE_POSITIVE_COMPONENTS",
           "FALSE_POSITIVE_VOLUME_ML", "LESION_POSITIVE", "NAUC_DICE", "NAUC_DMM", "ROUND_METRICS", "RoundScores",
           "binary_mask", "dice", "nauc", "score_examination", "score_round"]

#: Connectivity of lesions and predicted components in the official evaluation: voxels that share a face or an edge.
CONNECTIVITY = 18
#: A predicted component matches a lesion when their intersection over union is at least this value (official DMM).
DMM_IOU_THRESHOLD = 0.1
#: Key under which the official evaluator files the scores of one call; one evaluator scores one mask.
OFFICIAL_CASE_NAME = "examination"

# Keys of the scores of one examination (score_examination). A per-round metric holds one value for each of rounds 0
# to N; round 0 is the initial segmentation.
#: Whether the ground truth holds at least one lesion voxel.
LESION_POSITIVE = "lesion_positive"
DICE = "dice"
DMM = "dmm"
FALSE_POSITIVE_COMPONENTS = "false_positive_components"
FALSE_POSITIVE_VOLUME_ML = "false_positive_volume_ml"
NAUC_DICE = "nauc_dice"
NAUC_DMM = "nauc_dmm"
#: Metrics with one value per round.
ROUND_METRICS = (DICE, DMM, FALSE_POSITIVE_COMPONENTS, FALSE_POSITIVE_VOLUME_ML)
#: Metrics with one value per examination.
EXAMINATION_METRICS = (NAUC_DICE, NAUC_DMM)


def _official() -> Any:
    """The official autoPET V metric module. It imports matplotlib and nibabel at module level, so it is loaded on
    first use rather than when this module is imported."""
    from sirb.third_party.autopet_v import metrics  # noqa: PLC0415

    return metrics


def binary_mask(array: Any, name: str = "mask") -> np.ndarray:
    """``array`` as a boolean 3D array; refuses other dimensions and values other than 0 and 1."""
    mask = np.asarray(array)
    if mask.ndim != 3:
        raise ValueError(f"{name} must be a 3D array, got {mask.ndim} dimensions")
    if mask.dtype != np.bool_:
        if not ((mask == 0) | (mask == 1)).all():
            raise ValueError(f"{name} must hold only 0 and 1")
        mask = mask.astype(bool)
    return mask


def _pair(prediction: Any, ground_truth: Any) -> tuple[np.ndarray, np.ndarray]:
    p, g = binary_mask(prediction, "prediction"), binary_mask(ground_truth, "ground_truth")
    if p.shape != g.shape:
        raise ValueError(f"prediction shape {p.shape} differs from ground truth shape {g.shape}")
    return p, g


def _defined(value: Any) -> float | None:
    """A score of the official module as a float; its NaN (undefined on empty ground truth) as None."""
    value = float(value)
    return None if math.isnan(value) else value


def _checked_voxel_volume(voxel_volume_ml: Any) -> float:
    if isinstance(voxel_volume_ml, (bool, np.bool_)):
        raise TypeError("voxel_volume_ml must be a number, got a boolean")
    volume = float(voxel_volume_ml)
    if not math.isfinite(volume) or volume <= 0:
        raise ValueError(f"voxel_volume_ml must be a positive finite number, got {voxel_volume_ml!r}")
    return volume


def dice(prediction: Any, ground_truth: Any) -> float | None:
    """Global Dice of the official module; None when the ground truth is empty."""
    p, g = _pair(prediction, ground_truth)
    return _defined(_official().calc_dice(p.astype(np.uint8), g.astype(np.uint8)))


@dataclass(frozen=True)
class RoundScores:
    """Scores of one mask (module docstring). ``dice`` and ``dmm`` are None when the ground truth is empty."""

    dice: float | None
    dmm: float | None
    false_positive_components: int
    false_positive_volume_ml: float


def score_round(prediction: Any, ground_truth: Any, *, voxel_volume_ml: float) -> RoundScores:
    """Dice, DMM, false-positive components and false-positive volume of one mask.

    ``voxel_volume_ml`` is the volume of one voxel of the native PET grid in mL, the absolute determinant of the
    3 x 3 block of the voxel-to-world affine divided by 1000."""
    p, g = _pair(prediction, ground_truth)
    volume = _checked_voxel_volume(voxel_volume_ml)
    evaluator = _official().MetricEvaluator(overlap_threshold=DMM_IOU_THRESHOLD, connectivity=CONNECTIVITY)
    result = evaluator(p.astype(np.uint8), g.astype(np.uint8), case_name=OFFICIAL_CASE_NAME)
    return RoundScores(dice=_defined(result["dsc"]), dmm=_defined(result["f1"]),
                       false_positive_components=int(result["fp"]),
                       false_positive_volume_ml=int(np.count_nonzero(p & ~g)) * volume)


def nauc(values: Sequence[float]) -> float:
    """Normalised area under the curve of the scores after rounds 0 to N:
    (1/N) * sum over t = 1..N of (q[t-1] + q[t]) / 2, the trapezoidal area over N unit steps divided by N.

    The N pairwise means are added with numpy's sum, as ``numpy.trapz`` adds them, so the value equals
    ``numpy.trapz(values) / N`` to the last bit; for N <= 7 numpy adds them left to right."""
    values = list(values)
    if any(value is None for value in values):
        raise ValueError("nAUC needs a score after every round, got None")
    q = np.asarray(values, dtype=np.float64)
    if q.ndim != 1 or q.size < 2:
        raise ValueError(f"nAUC needs the scores of at least two rounds, got shape {q.shape}")
    if not np.isfinite(q).all():
        raise ValueError(f"nAUC needs finite scores, got {q.tolist()}")
    return float(((q[1:] + q[:-1]) / 2.0).sum() / (q.size - 1))


def score_examination(masks: Sequence[Any], ground_truth: Any, *, voxel_volume_ml: float) -> dict[str, Any]:
    """Scores of the masks after rounds 0 to N of one examination; ``masks[0]`` is the initial segmentation.

    Returns a JSON-ready dictionary: LESION_POSITIVE; one value per round for each of ROUND_METRICS; NAUC_DICE and
    NAUC_DMM over all rounds. On empty ground truth the Dice, DMM and nAUC entries are None."""
    if len(masks) < 2:
        raise ValueError(f"an examination needs the masks of at least two rounds, got {len(masks)}")
    g = binary_mask(ground_truth, "ground_truth")
    rounds = [score_round(mask, g, voxel_volume_ml=voxel_volume_ml) for mask in masks]
    positive = bool(g.any())
    dice_curve = [scores.dice for scores in rounds]
    dmm_curve = [scores.dmm for scores in rounds]
    return {
        LESION_POSITIVE: positive,
        DICE: dice_curve,
        DMM: dmm_curve,
        FALSE_POSITIVE_COMPONENTS: [scores.false_positive_components for scores in rounds],
        FALSE_POSITIVE_VOLUME_ML: [scores.false_positive_volume_ml for scores in rounds],
        NAUC_DICE: nauc(dice_curve) if positive else None,
        NAUC_DMM: nauc(dmm_curve) if positive else None,
    }
