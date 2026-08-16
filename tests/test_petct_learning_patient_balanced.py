from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


PROJECT = Path(__file__).resolve().parents[1]
SCRIPTS = PROJECT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from common.petct_learning import (  # noqa: E402
    patient_balanced_macro_f1,
    patient_balanced_macro_f1_summary,
    patient_cluster_bootstrap_macro_f1,
)


def _reference_d14(y_true, y_pred, patient_ids, labels) -> float:
    grouped: dict[str, list[int]] = {}
    for index, patient in enumerate(patient_ids):
        grouped.setdefault(str(patient), []).append(index)
    class_scores = []
    for label in labels:
        patient_scores = []
        for indices in grouped.values():
            tp = sum(
                y_true[index] == label and y_pred[index] == label
                for index in indices
            )
            fp = sum(
                y_true[index] != label and y_pred[index] == label
                for index in indices
            )
            fn = sum(
                y_true[index] == label and y_pred[index] != label
                for index in indices
            )
            denominator = 2 * tp + fp + fn
            if denominator:
                patient_scores.append(2.0 * tp / denominator)
        if patient_scores:
            class_scores.append(float(np.mean(patient_scores)))
    return float(np.mean(class_scores))


def _retired_inverse_episode_weighted_f1(y_true, y_pred, patient_ids, labels) -> float:
    episode_counts: dict[str, int] = {}
    for patient in patient_ids:
        episode_counts[str(patient)] = episode_counts.get(str(patient), 0) + 1
    weights = [1.0 / episode_counts[str(patient)] for patient in patient_ids]
    class_scores = []
    for label in labels:
        tp = sum(
            weight
            for truth, prediction, weight in zip(y_true, y_pred, weights)
            if truth == label and prediction == label
        )
        fp = sum(
            weight
            for truth, prediction, weight in zip(y_true, y_pred, weights)
            if truth != label and prediction == label
        )
        fn = sum(
            weight
            for truth, prediction, weight in zip(y_true, y_pred, weights)
            if truth == label and prediction != label
        )
        denominator = 2.0 * tp + fp + fn
        class_scores.append(0.0 if denominator == 0 else 2.0 * tp / denominator)
    return float(np.mean(class_scores))


def test_d14_counterexample_differs_from_retired_inverse_episode_weighting() -> None:
    y_true = [0] * 101
    y_pred = [0] * 100 + [1]
    patient_ids = ["patient-a"] * 100 + ["patient-b"]

    summary = patient_balanced_macro_f1_summary(
        y_true, y_pred, patient_ids, labels=[0, 1]
    )

    score = patient_balanced_macro_f1(
        y_true, y_pred, patient_ids, labels=[0, 1]
    )
    assert isinstance(score, float)
    assert _retired_inverse_episode_weighted_f1(
        y_true, y_pred, patient_ids, labels=[0, 1]
    ) == pytest.approx(1.0 / 3.0)
    assert score == pytest.approx(0.25)
    assert summary["estimate"] == pytest.approx(0.25)
    assert summary["per_class_f1"] == pytest.approx({"0": 0.5, "1": 0.0})
    assert summary["per_class_support"] == {"0": 2, "1": 1}
    assert summary["defined_class_count"] == 2


def test_undefined_patient_class_cells_and_globally_undefined_class_are_skipped() -> None:
    summary = patient_balanced_macro_f1_summary(
        y_true=[0, 0],
        y_pred=[0, 0],
        patient_ids=["patient-a", "patient-b"],
        labels=[0, 1],
    )

    assert summary["estimate"] == pytest.approx(1.0)
    assert summary["per_class_f1"] == {"0": 1.0, "1": None}
    assert summary["per_class_support"] == {"0": 2, "1": 0}
    assert summary["defined_class_count"] == 1


def test_duplicating_episodes_within_one_patient_does_not_change_the_estimate() -> None:
    y_true = [0, 0, 1, 0, 1]
    y_pred = [0, 1, 1, 1, 1]
    patient_ids = ["patient-a"] * 3 + ["patient-b"] * 2
    baseline = patient_balanced_macro_f1(y_true, y_pred, patient_ids, labels=[0, 1])

    repeated_truth = y_true[:3] * 7 + y_true[3:]
    repeated_prediction = y_pred[:3] * 7 + y_pred[3:]
    repeated_patients = ["patient-a"] * 21 + ["patient-b"] * 2

    assert baseline == pytest.approx(0.5)
    assert patient_balanced_macro_f1(
        repeated_truth, repeated_prediction, repeated_patients, labels=[0, 1]
    ) == pytest.approx(baseline)


def test_cluster_bootstrap_resamples_patients_and_recomputes_the_d14_estimand() -> None:
    y_true = [0, 0, 1, 0, 1, 1]
    y_pred = [0, 1, 1, 0, 0, 1]
    patient_ids = ["patient-a"] * 3 + ["patient-b"] * 2 + ["patient-c"]
    labels = [0, 1]
    seed = 20260814
    samples = 101

    grouped: dict[str, list[int]] = {}
    for index, patient in enumerate(patient_ids):
        grouped.setdefault(patient, []).append(index)
    patients = sorted(grouped)
    rng = np.random.default_rng(seed)
    reference_draws = []
    for _ in range(samples):
        selected = rng.choice(patients, size=len(patients), replace=True)
        draw_truth, draw_prediction, draw_patients = [], [], []
        for draw_index, patient in enumerate(selected):
            for row_index in grouped[str(patient)]:
                draw_truth.append(y_true[row_index])
                draw_prediction.append(y_pred[row_index])
                draw_patients.append(f"{draw_index}:{patient}")
        reference_draws.append(
            _reference_d14(draw_truth, draw_prediction, draw_patients, labels)
        )

    result = patient_cluster_bootstrap_macro_f1(
        y_true,
        y_pred,
        patient_ids,
        labels,
        seed=seed,
        samples=samples,
        alpha=0.05,
    )

    point_summary = patient_balanced_macro_f1_summary(
        y_true, y_pred, patient_ids, labels
    )
    assert result["estimate"] == pytest.approx(
        _reference_d14(y_true, y_pred, patient_ids, labels)
    )
    assert result["ci_low"] == pytest.approx(np.quantile(reference_draws, 0.025))
    assert result["ci_high"] == pytest.approx(np.quantile(reference_draws, 0.975))
    assert result["per_class_f1"] == point_summary["per_class_f1"]
    assert result["per_class_support"] == point_summary["per_class_support"]
