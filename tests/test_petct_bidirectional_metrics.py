from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "scripts" / "evaluation"))

from petct_bidirectional_metrics import (  # noqa: E402
    BidirectionalMetricError,
    bidirectional_correction_metrics,
)


def _base():
    gt = np.zeros((5, 5), dtype=np.uint8)
    gt[1, 1] = 1
    gt[3, 3] = 1
    m0 = np.zeros_like(gt)
    m0[1, 1] = 1
    m0[2, 2] = 1
    return gt, m0


def test_add_uses_union_and_scores_unauthorized_addition() -> None:
    gt, m0 = _base()
    authorized = np.zeros_like(gt)
    authorized[3, 3] = 1
    cue = authorized.copy()
    m1 = m0.copy()
    m1[3, 3] = 1
    m1[4, 4] = 1
    result = bidirectional_correction_metrics(
        gt=gt,
        m0=m0,
        m1=m1,
        authorized_target=authorized,
        cue_support=cue,
        operation="ADD",
        scope="LOCAL",
        spacing_xyz=(1.0, 1.0),
        distal_radius_mm=1.0,
    )
    assert result["authorized_add_recall"] == 1.0
    assert result["authorized_remove_recall"] is None
    assert result["unauthorized_addition_voxels"] == 1.0
    assert result["unauthorized_removal_voxels"] == 0.0
    assert result["true_positive_preservation_rate"] == 1.0


def test_remove_uses_subtraction_and_protects_true_positive_voxels() -> None:
    gt, m0 = _base()
    authorized = np.zeros_like(gt)
    authorized[2, 2] = 1
    cue = authorized.copy()
    m1 = m0.copy()
    m1[2, 2] = 0
    result = bidirectional_correction_metrics(
        gt=gt,
        m0=m0,
        m1=m1,
        authorized_target=authorized,
        cue_support=cue,
        operation="REMOVE",
        scope="COMPLETE",
        spacing_xyz=(1.0, 1.0),
        distal_radius_mm=1.0,
    )
    assert result["authorized_remove_recall"] == 1.0
    assert result["unauthorized_removal_voxels"] == 0.0
    assert result["true_positive_preservation_rate"] == 1.0
    assert result["complete_fp_removal"] == 1.0
    assert result["complete_fp_removal_defined"] is True
    assert result["delta_dice"] > 0


def test_remove_reports_unauthorized_true_positive_removal() -> None:
    gt, m0 = _base()
    authorized = np.zeros_like(gt)
    authorized[2, 2] = 1
    cue = authorized.copy()
    m1 = np.zeros_like(m0)
    result = bidirectional_correction_metrics(
        gt=gt,
        m0=m0,
        m1=m1,
        authorized_target=authorized,
        cue_support=cue,
        operation="REMOVE",
        scope="LOCAL",
        spacing_xyz=(1.0, 1.0),
        distal_radius_mm=1.0,
    )
    assert result["unauthorized_removal_voxels"] == 1.0
    assert result["true_positive_preservation_rate"] == 0.0


def test_operation_algebra_and_authorized_domain_fail_closed() -> None:
    gt, m0 = _base()
    authorized = np.zeros_like(gt)
    authorized[3, 3] = 1
    with pytest.raises(BidirectionalMetricError, match="violates"):
        bidirectional_correction_metrics(
            gt=gt,
            m0=m0,
            m1=np.zeros_like(m0),
            authorized_target=authorized,
            cue_support=authorized,
            operation="ADD",
            scope="LOCAL",
            spacing_xyz=(1.0, 1.0),
            distal_radius_mm=1.0,
        )
    with pytest.raises(BidirectionalMetricError, match="outside"):
        bidirectional_correction_metrics(
            gt=gt,
            m0=m0,
            m1=m0,
            authorized_target=authorized,
            cue_support=authorized,
            operation="REMOVE",
            scope="LOCAL",
            spacing_xyz=(1.0, 1.0),
            distal_radius_mm=1.0,
        )


def test_predicted_wrong_operation_is_valid_execution_but_fails_target_safety() -> None:
    gt = np.asarray([[1, 0], [0, 0]], dtype=np.uint8)
    m0 = np.asarray([[1, 0], [1, 0]], dtype=np.uint8)
    authorized_remove = np.asarray([[0, 0], [1, 0]], dtype=np.uint8)
    # End-to-end P2T predicted ADD on a gold REMOVE episode.
    m1 = np.asarray([[1, 1], [1, 0]], dtype=np.uint8)
    metrics = bidirectional_correction_metrics(
        gt=gt,
        m0=m0,
        m1=m1,
        authorized_target=authorized_remove,
        cue_support=authorized_remove,
        operation="REMOVE",
        execution_operation="ADD",
        scope="LOCAL",
        spacing_xyz=(1.0, 1.0),
        distal_radius_mm=0.0,
    )
    assert metrics["execution_operation"] == "ADD"
    assert metrics["execution_operation_matches_target"] is False
    assert metrics["target_operation_safety_pass"] == 0.0
    assert metrics["authorized_remove_recall"] == 0.0
    assert metrics["unauthorized_addition_voxels"] == 1.0
