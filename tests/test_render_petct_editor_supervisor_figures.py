from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS / "evaluation"))

from render_petct_editor_supervisor_figures import (  # noqa: E402
    FigureContractError,
    _binary,
    _case_panels,
    _dice,
    _inverse_crop_binary,
    _ladder_summary,
    _roi_box,
    _target_recovery,
)


def test_case_panels_compare_the_same_editor_result_with_and_without_overlay() -> None:
    gt = np.zeros((8, 8), dtype=bool)
    m0 = np.zeros_like(gt)
    m1 = np.ones_like(gt)
    added = m1.copy()
    removed = np.zeros_like(gt)
    scribble = ((3, 4), (4, 4))
    case = SimpleNamespace(
        gt=gt,
        m0=m0,
        m1=m1,
        editor_added=added,
        editor_removed=removed,
        scribble_points=scribble,
    )

    panels = _case_panels(case)
    assert [panel[0] for panel in panels] == [
        "GROUND TRUTH",
        "M0 · INITIAL",
        "EDITOR RESULT",
        "EDITOR RESULT + SCRIBBLE",
    ]
    assert panels[2][1] is m1
    assert panels[3][1] is m1
    assert panels[2][2:4] == panels[3][2:4]
    assert panels[2][4] is None
    assert panels[3][4] == scribble


def test_binary_contract_and_dice() -> None:
    a = np.zeros((256, 256), dtype=np.uint8)
    b = np.zeros_like(a)
    a[10:20, 10:20] = 1
    b[15:25, 10:20] = 1
    assert _binary(a, name="a").dtype == np.bool_
    assert _dice(a > 0, b > 0) == pytest.approx(0.5)
    assert _dice(np.zeros_like(a, dtype=bool), np.zeros_like(a, dtype=bool)) == 1.0
    with pytest.raises(FigureContractError, match=r"256, 256"):
        _binary(np.zeros((32, 32)), name="bad")
    a[0, 0] = 2
    with pytest.raises(FigureContractError, match="binary"):
        _binary(a, name="bad")


def test_inverse_crop_binary_identity_geometry() -> None:
    crop = np.zeros((8, 8), dtype=np.uint8)
    crop[3:5, 2:6] = 1
    restored = _inverse_crop_binary(
        crop,
        output_shape=(8, 8),
        center_xy=(3.5, 3.5),
        original_spacing_xy=(1.0, 1.0),
        field_mm=8.0,
    )
    assert np.array_equal(restored, crop > 0)


def test_roi_box_is_square_and_contains_the_edit() -> None:
    mask = np.zeros((200, 180), dtype=bool)
    mask[80:90, 70:95] = True
    box = _roi_box((mask,), min_side=64, padding=8)
    x0, x1, y0, y1 = box
    assert x1 - x0 == y1 - y0 == 64
    assert x0 <= 80 < 90 <= x1
    assert y0 <= 70 < 95 <= y1


def test_target_recovery_is_recall_not_change_precision() -> None:
    authorized = np.zeros((16, 16), dtype=bool)
    authorized[4:8, 4:8] = True
    changed = np.zeros_like(authorized)
    changed[4:6, 4:8] = True
    changed[12:14, 12:14] = True  # outside-target change must not inflate recall
    recovery, denominator, total_changed = _target_recovery(changed, authorized)
    assert recovery == pytest.approx(0.5)
    assert denominator == 16
    assert total_changed == 12


def test_ladder_summary_uses_only_observed_rounds(tmp_path: Path) -> None:
    rows = [
        {
            "d0": 0.5,
            "d2d_r1": 0.6,
            "d3d_r1": 0.7,
            "d2d_r2": 0.65,
            "d3d_r2": 0.8,
        },
        {
            "d0": 0.7,
            "d2d_r1": 0.8,
            "d3d_r1": 0.9,
            "d2d_r2": None,
            "d3d_r2": None,
        },
    ]
    for row in rows:
        for index in range(3, 6):
            row[f"d2d_r{index}"] = row.get("d2d_r2")
            row[f"d3d_r{index}"] = row.get("d3d_r2")
    path = tmp_path / "rows.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    two_d, three_d, n_two, n_three = _ladder_summary(path)
    assert two_d[:3] == pytest.approx([0.6, 0.7, 0.65])
    assert three_d[:3] == pytest.approx([0.6, 0.8, 0.8])
    assert n_two[:3] == [2, 2, 1]
    assert n_three[:3] == [2, 2, 1]
