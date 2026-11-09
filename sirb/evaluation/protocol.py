"""Evaluation protocol: per-examination scores to summaries averaged by patient and by examination.

The patient is the independent unit. A row holds the scores of one examination under one scribble style: the output
of ``sirb.evaluation.metrics.score_examination`` with the identifiers ``examination``, ``patient`` and ``style``.

* An examination value is the mean of the examination's rows, one per style.
* A patient value is the mean of the patient's examination values.
* Averaged by patient: the mean of the patient values, every patient with equal weight.
* Standard deviation across patients: the sample standard deviation (ddof 1) of the patient values.
* Examination level: every examination with equal weight, the sum of the examination values over their number.
* Bootstrap interval (optional, ``Bootstrap``): the percentile interval of an estimate over patient-cluster
  resamples. Each resample draws as many patients as there are, with replacement, from
  ``numpy.random.default_rng(seed)`` and takes all of their examinations.

Examinations with empty ground truth have no Dice, DMM or nAUC. They are left out of every summary and counted under
EMPTY_GROUND_TRUTH, so every metric is summarised over the same lesion-positive examinations.

The order of summation is fixed, so the numbers do not depend on the order of the rows except for the styles:
patients in identifier order; a patient's examinations in identifier order, added left to right; the styles of an
examination in the order of the rows. The numpy expressions below (means, sample standard deviation, resampling and
percentiles) are part of that definition: replacing one with an equivalent formula can change the last bit.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from numbers import Integral, Real
from typing import Any, Iterable, Mapping

import numpy as np

from sirb.evaluation.metrics import (DICE, EXAMINATION_METRICS, FALSE_POSITIVE_COMPONENTS, FALSE_POSITIVE_VOLUME_ML,
                                     LESION_POSITIVE, NAUC_DICE, NAUC_DMM, ROUND_METRICS)

__all__ = ["EMPTY_GROUND_TRUTH", "EXAMINATION", "FINAL", "LEVELS", "PATIENT", "STYLE", "SUMMARY_METRICS", "Bootstrap",
           "ExaminationValues", "PatientValues", "Summary", "examination_values", "round_summaries", "summarize",
           "summary_table"]

#: Row keys of the identifiers; PATIENT and EXAMINATION also name the two summary levels.
EXAMINATION = "examination"
PATIENT = "patient"
STYLE = "style"
#: Summary levels: averaged by patient, or every examination with equal weight.
LEVELS = (PATIENT, EXAMINATION)
#: Why an examination is left out of the summaries: its ground truth is empty.
EMPTY_GROUND_TRUTH = "empty ground truth"
#: Index of the final round.
FINAL = -1
#: Name, metric and round of the summaries of the overall results.
SUMMARY_METRICS = (("Final Dice", DICE, FINAL),
                   ("nAUC-Dice", NAUC_DICE, None),
                   ("nAUC-DMM", NAUC_DMM, None),
                   ("Final false-positive components", FALSE_POSITIVE_COMPONENTS, FINAL),
                   ("Final false-positive volume", FALSE_POSITIVE_VOLUME_ML, FINAL))


@dataclass(frozen=True)
class Bootstrap:
    """Settings of the patient-cluster bootstrap interval: ``resamples`` resamples drawn with
    ``numpy.random.default_rng(seed)``; the interval holds the central ``confidence`` share of the resampled
    estimates. No setting has a default: they come from the evaluation configuration."""

    resamples: int
    seed: int
    confidence: float

    def __post_init__(self) -> None:
        for name in ("resamples", "seed"):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
                raise TypeError(f"bootstrap {name} must be an integer, got {value!r}")
            object.__setattr__(self, name, int(value))
        if self.resamples < 1 or self.seed < 0:
            raise ValueError(f"bootstrap needs at least one resample and a non-negative seed, got {self}")
        confidence = self.confidence
        if isinstance(confidence, (bool, np.bool_)) or not isinstance(confidence, Real):
            raise TypeError(f"bootstrap confidence must be a number, got {confidence!r}")
        if not 0 < confidence < 1:
            raise ValueError(f"bootstrap confidence must lie strictly between 0 and 1, got {confidence!r}")
        object.__setattr__(self, "confidence", float(confidence))


def _check_level(level: str) -> None:
    if level not in LEVELS:
        raise ValueError(f"summary level must be one of {LEVELS}, got {level!r}")


def _left_to_right_sum(values: Iterable[float]) -> float:
    """Sum of floats added one by one from the left (Python's ``sum`` adds with compensation since 3.12)."""
    total = 0.0
    for value in values:
        total += value
    return total


class PatientValues:
    """Examination values grouped by patient (module docstring).

    ``patients`` are the patient identifiers in order; ``sums``, ``counts`` and ``means`` are, per patient, the sum of
    the examination values added left to right in examination order, the number of examinations and the patient
    value."""

    def __init__(self, values: Mapping[tuple[str, str], float]):
        by_patient: dict[str, list[float]] = {}
        for (patient, _examination), value in sorted(values.items()):
            by_patient.setdefault(patient, []).append(float(value))
        self.patients = tuple(sorted(by_patient))
        self.sums = np.array([_left_to_right_sum(by_patient[patient]) for patient in self.patients], dtype=float)
        self.counts = np.array([len(by_patient[patient]) for patient in self.patients], dtype=float)
        self.means = self.sums / self.counts

    @property
    def n_patients(self) -> int:
        return len(self.patients)

    @property
    def n_examinations(self) -> int:
        return int(self.counts.sum())

    def as_dict(self) -> dict[str, float]:
        """Patient identifier to patient value."""
        return {patient: float(value) for patient, value in zip(self.patients, self.means)}

    def mean(self, level: str) -> float | None:
        """Averaged by patient (PATIENT) or every examination with equal weight (EXAMINATION); None without patients."""
        _check_level(level)
        if not self.patients:
            return None
        if level == PATIENT:
            return float(self.means.mean())
        return float(self.sums.sum() / self.counts.sum())

    def standard_deviation(self) -> float | None:
        """Standard deviation across patients (ddof 1); None with fewer than two patients."""
        return float(self.means.std(ddof=1)) if self.n_patients >= 2 else None

    def bootstrap_interval(self, level: str, bootstrap: Bootstrap) -> tuple[float, float] | None:
        """Percentile interval of ``mean(level)`` over patient-cluster resamples; None with fewer than two patients."""
        _check_level(level)
        if not isinstance(bootstrap, Bootstrap):
            raise TypeError(f"bootstrap must be a Bootstrap, got {type(bootstrap).__name__}")
        if self.n_patients < 2:
            return None
        rng = np.random.default_rng(bootstrap.seed)
        index = rng.integers(0, self.n_patients, size=(bootstrap.resamples, self.n_patients))
        if level == PATIENT:
            draws = self.means[index].mean(axis=1)
        else:
            draws = self.sums[index].sum(axis=1) / self.counts[index].sum(axis=1)
        tail = (1 - bootstrap.confidence) / 2 * 100
        low, high = np.percentile(draws, [tail, 100 - tail])
        return float(low), float(high)


@dataclass(frozen=True)
class ExaminationValues:
    """The value of one metric for each lesion-positive examination, keyed by (patient, examination); the
    examinations left out, by reason; the numbers of patients and examinations among all rows; and the round
    (non-negative index; None for a metric with one value per examination)."""

    values: dict[tuple[str, str], float]
    excluded: dict[str, tuple[str, ...]]
    n_patients_total: int
    n_examinations_total: int
    round: int | None


def _round_count(rows: list[Mapping[str, Any]], metric: str) -> int | None:
    """The number of rounds (0 to N) of a per-round metric, common to every row; None without rows."""
    counts = {len(row[metric]) for row in rows}
    if len(counts) > 1:
        raise ValueError(f"rows hold {metric} for different numbers of rounds: {sorted(counts)}")
    return counts.pop() if counts else None


def _round_index(rows: list[Mapping[str, Any]], metric: str, round: int | None) -> int | None:
    if metric in EXAMINATION_METRICS:
        if round is not None:
            raise ValueError(f"{metric} has one value per examination; round must be None, got {round!r}")
        return None
    if metric not in ROUND_METRICS:
        raise ValueError(f"unknown metric {metric!r}; known: {ROUND_METRICS + EXAMINATION_METRICS}")
    if round is None:
        raise ValueError(f"{metric} has one value per round; give the round (FINAL for the last)")
    if isinstance(round, (bool, np.bool_)) or not isinstance(round, Integral):
        raise TypeError(f"round must be an integer, got {round!r}")
    count = _round_count(rows, metric)
    if count is None:
        return int(round)
    if not -count <= round < count:
        raise ValueError(f"round {round} is outside rounds 0 to {count - 1}")
    return int(round) % count


def _value(row: Mapping[str, Any], metric: str, index: int | None) -> float:
    value = row[metric] if index is None else row[metric][index]
    if value is None:
        raise ValueError(f"lesion-positive examination {row[EXAMINATION]!r} has no {metric}")
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{metric} of examination {row[EXAMINATION]!r} must be a number, got {value!r}")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{metric} of examination {row[EXAMINATION]!r} is not finite: {value}")
    return value


def examination_values(rows: Iterable[Mapping[str, Any]], metric: str, *, round: int | None = None) \
        -> ExaminationValues:
    """The value of ``metric`` for each lesion-positive examination, averaged over its styles in the order of the rows.

    ``round`` selects the round of a per-round metric (FINAL for the last) and must be None for a metric with one value
    per examination. Refuses rows that give an examination two patients, disagree on whether its ground truth holds a
    lesion, repeat a style of an examination, or lack a finite value on a lesion-positive examination."""
    rows = list(rows)
    index = _round_index(rows, metric, round)
    patient_of: dict[str, str] = {}
    positive_of: dict[str, bool] = {}
    styles: set[tuple[str, Any]] = set()
    by_examination: dict[tuple[str, str], list[float]] = {}
    for row in rows:
        examination, patient, style, positive = row[EXAMINATION], row[PATIENT], row[STYLE], row[LESION_POSITIVE]
        if not isinstance(positive, (bool, np.bool_)):
            raise TypeError(f"{LESION_POSITIVE} of examination {examination!r} must be a boolean, got {positive!r}")
        if patient_of.setdefault(examination, patient) != patient:
            raise ValueError(f"examination {examination!r} belongs to patient {patient_of[examination]!r} and to "
                             f"patient {patient!r}")
        if positive_of.setdefault(examination, bool(positive)) != bool(positive):
            raise ValueError(f"the rows of examination {examination!r} disagree on whether its ground truth holds a "
                             f"lesion")
        if (examination, style) in styles:
            raise ValueError(f"examination {examination!r} has style {style!r} twice")
        styles.add((examination, style))
        if positive:
            by_examination.setdefault((patient, examination), []).append(_value(row, metric, index))
    empty = tuple(sorted(examination for examination, positive in positive_of.items() if not positive))
    return ExaminationValues(values={key: float(np.mean(values)) for key, values in by_examination.items()},
                             excluded={EMPTY_GROUND_TRUTH: empty}, n_patients_total=len(set(patient_of.values())),
                             n_examinations_total=len(patient_of), round=index)


@dataclass(frozen=True)
class Summary:
    """Summary of one metric (at one round) over the lesion-positive examinations (module docstring).

    ``excluded`` gives the number of examinations left out by reason; the ``_total`` counts include them. The
    intervals are None without a bootstrap or with fewer than two patients."""

    metric: str
    round: int | None
    patient_mean: float | None
    patient_sd: float | None
    examination_mean: float | None
    n_patients: int
    n_examinations: int
    n_patients_total: int
    n_examinations_total: int
    excluded: dict[str, int]
    bootstrap: Bootstrap | None
    patient_interval: tuple[float, float] | None
    examination_interval: tuple[float, float] | None

    def to_dict(self) -> dict[str, Any]:
        """The summary as a JSON-ready dictionary."""
        return asdict(self)


def summarize(rows: Iterable[Mapping[str, Any]], metric: str, *, round: int | None = None,
              bootstrap: Bootstrap | None = None) -> Summary:
    """Summary of ``metric`` (at ``round``, as in ``examination_values``), with bootstrap intervals when
    ``bootstrap`` is given."""
    if bootstrap is not None and not isinstance(bootstrap, Bootstrap):
        raise TypeError(f"bootstrap must be a Bootstrap or None, got {type(bootstrap).__name__}")
    selected = examination_values(rows, metric, round=round)
    patients = PatientValues(selected.values)
    intervals = {level: None if bootstrap is None else patients.bootstrap_interval(level, bootstrap)
                 for level in LEVELS}
    return Summary(metric=metric, round=selected.round, patient_mean=patients.mean(PATIENT),
                   patient_sd=patients.standard_deviation(), examination_mean=patients.mean(EXAMINATION),
                   n_patients=patients.n_patients, n_examinations=patients.n_examinations,
                   n_patients_total=selected.n_patients_total, n_examinations_total=selected.n_examinations_total,
                   excluded={reason: len(examinations) for reason, examinations in selected.excluded.items()},
                   bootstrap=bootstrap, patient_interval=intervals[PATIENT],
                   examination_interval=intervals[EXAMINATION])


def round_summaries(rows: Iterable[Mapping[str, Any]], metric: str, *, bootstrap: Bootstrap | None = None) \
        -> list[Summary]:
    """One summary of a per-round metric for each of rounds 0 to N: the curve averaged by patient."""
    rows = list(rows)
    if metric not in ROUND_METRICS:
        raise ValueError(f"{metric!r} is not a per-round metric; per-round metrics: {ROUND_METRICS}")
    return [summarize(rows, metric, round=index, bootstrap=bootstrap)
            for index in range(_round_count(rows, metric) or 0)]


def summary_table(rows: Iterable[Mapping[str, Any]], *, bootstrap: Bootstrap | None = None) -> dict[str, Summary]:
    """The summaries of SUMMARY_METRICS, by name."""
    rows = list(rows)
    return {name: summarize(rows, metric, round=index, bootstrap=bootstrap) for name, metric, index in SUMMARY_METRICS}
