from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(SCRIPTS / "evaluation"))

from render_petct_intent_montage import (  # noqa: E402
    MontageContractError,
    render_petct_intent_montage,
)


def _inputs() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    shape = (32, 32, 5)
    ct = np.linspace(-1000, 500, np.prod(shape), dtype=np.float32).reshape(shape)
    pet = np.zeros(shape, dtype=np.float32)
    pet[8:24, 8:24, :] = np.linspace(0.5, 12.0, 16 * 16 * 5).reshape(16, 16, 5)
    m0 = np.zeros(shape, dtype=np.uint8)
    m0[10:15, 10:15, 2] = 1
    scribble = np.zeros(shape, dtype=np.uint8)
    scribble[18:21, 18, 2] = 1
    return pet, ct, m0, scribble


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_montage_is_deterministic_blind_2p5d_packet(tmp_path: Path) -> None:
    pet, ct, m0, scribble = _inputs()
    first_path = tmp_path / "first" / "ep-000001.png"
    second_path = tmp_path / "second" / "ep-000001.png"

    first = render_petct_intent_montage(
        pet,
        ct,
        m0,
        scribble,
        episode_id="ep-000001",
        output_path=first_path,
        tile_size=64,
    )
    second = render_petct_intent_montage(
        pet,
        ct,
        m0,
        scribble,
        episode_id="ep-000001",
        output_path=second_path,
        tile_size=64,
    )

    assert first["status"] == "COMMITTED"
    assert first["view"] == "axial-overview-2p5d"
    assert first["center_slice"] == 2
    assert first["slice_offsets"] == [-2, -1, 0, 1, 2]
    assert first["slice_indices"] == [0, 1, 2, 3, 4]
    assert first["ct_window_hu"] == [-150.0, 250.0]
    assert first["pet_scale"]["method"] == "volume-positive-p99.5"
    assert first["contains_gt"] is False
    assert first["contains_gold_intent"] is False
    assert first["sha256"] == _sha(first_path)
    assert first_path.read_bytes() == second_path.read_bytes()
    assert first["sha256"] == second["sha256"]

    image = Image.open(first_path).convert("RGB")
    assert image.size == (5 * 64, 32 + 3 * 64 + 24)
    colors = set(image.getdata())
    assert (255, 0, 255) in colors  # fixed foreground scribble overlay
    assert (0, 255, 255) in colors  # fixed current-M0 contour


def test_montage_rejects_scribble_on_multiple_slices(tmp_path: Path) -> None:
    pet, ct, m0, scribble = _inputs()
    scribble[18, 18, 1] = 1

    with pytest.raises(MontageContractError, match="single axial slice"):
        render_petct_intent_montage(
            pet,
            ct,
            m0,
            scribble,
            episode_id="ep-000001",
            output_path=tmp_path / "ep-000001.png",
        )


def test_montage_rejects_scribble_that_overlaps_m0(tmp_path: Path) -> None:
    pet, ct, m0, scribble = _inputs()
    scribble[:] = 0
    scribble[12, 12, 2] = 1

    with pytest.raises(MontageContractError, match="outside current M0"):
        render_petct_intent_montage(
            pet,
            ct,
            m0,
            scribble,
            episode_id="ep-000001",
            output_path=tmp_path / "ep-000001.png",
        )


def test_montage_rejects_nonfinite_or_misaligned_inputs(tmp_path: Path) -> None:
    pet, ct, m0, scribble = _inputs()
    pet[0, 0, 0] = np.nan
    with pytest.raises(MontageContractError, match="finite"):
        render_petct_intent_montage(
            pet,
            ct,
            m0,
            scribble,
            episode_id="ep-000001",
            output_path=tmp_path / "nan.png",
        )

    pet, ct, m0, scribble = _inputs()
    with pytest.raises(MontageContractError, match="shape"):
        render_petct_intent_montage(
            pet[:, :, :-1],
            ct,
            m0,
            scribble,
            episode_id="ep-000002",
            output_path=tmp_path / "shape.png",
        )


def test_montage_is_no_clobber_and_uses_opaque_episode_id(tmp_path: Path) -> None:
    pet, ct, m0, scribble = _inputs()
    output = tmp_path / "ep-000001.png"
    render_petct_intent_montage(
        pet,
        ct,
        m0,
        scribble,
        episode_id="ep-000001",
        output_path=output,
    )

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        render_petct_intent_montage(
            pet,
            ct,
            m0,
            scribble,
            episode_id="ep-000001",
            output_path=output,
        )
    with pytest.raises(MontageContractError, match="opaque"):
        render_petct_intent_montage(
            pet,
            ct,
            m0,
            scribble,
            episode_id="psma_patient_secret_2020",
            output_path=tmp_path / "bad.png",
        )


def test_edge_center_uses_blank_padding_not_duplicate_slices(tmp_path: Path) -> None:
    pet, ct, m0, scribble = _inputs()
    scribble[:] = 0
    scribble[18:21, 18, 0] = 1
    m0[:] = 0

    receipt = render_petct_intent_montage(
        pet,
        ct,
        m0,
        scribble,
        episode_id="ep-000003",
        output_path=tmp_path / "ep-000003.png",
        tile_size=32,
    )

    assert receipt["slice_indices"] == [None, None, 0, 1, 2]
    assert receipt["padded_offsets"] == [-2, -1]
