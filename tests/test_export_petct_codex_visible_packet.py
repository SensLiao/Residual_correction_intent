from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

import pytest
from PIL import Image


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(SCRIPTS / "orchestration"))

from export_petct_codex_visible_packet import (  # noqa: E402
    PacketContractError,
    export_visible_packet,
)


def _write_episode(root: Path, episode_id: str, *, polarity: str = "foreground") -> None:
    root.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "PETCT-EPISODE-VISIBLE-v2.0",
        "episode_id": episode_id,
        "lane": "controlled",
        "patient_group_hash": "a" * 64,
        "montage_reference": f"montages/{episode_id}.png",
        "m0_provenance": {"kind": "controlled", "operator_version": "v1"},
        "scribble": {
            "polarity": polarity,
            "strategy": "centerline",
            "seed": 42,
            "coordinate_count": 3,
            "coordinate_sha256": "b" * 64,
            "fallback_mode": "sparse",
        },
        "expected_model_output_schema": "PETCT-INTENT-v2.0",
    }
    (root / f"{episode_id}.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )


def _write_montage(root: Path, episode_id: str) -> None:
    root.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 32), color=(10, 20, 30)).save(
        root / f"{episode_id}.png", format="PNG"
    )


def test_visible_packet_is_deterministic_and_contains_only_allowlisted_planes(
    tmp_path: Path,
) -> None:
    visible_root = tmp_path / "visible"
    montage_root = tmp_path / "montages"
    for episode_id in ("ep-000001", "ep-000002"):
        _write_episode(visible_root, episode_id)
        _write_montage(montage_root, episode_id)
    first_path = tmp_path / "first" / "pilot3-visible-v1.zip"
    second_path = tmp_path / "second" / "pilot3-visible-v1.zip"

    first = export_visible_packet(
        visible_root=visible_root,
        montage_root=montage_root,
        protocol_path=PROJECT / "protocols" / "petct_codex_prompt_v2.json",
        output_zip=first_path,
        packet_id="pilot3-visible-v1",
    )
    second = export_visible_packet(
        visible_root=visible_root,
        montage_root=montage_root,
        protocol_path=PROJECT / "protocols" / "petct_codex_prompt_v2.json",
        output_zip=second_path,
        packet_id="pilot3-visible-v1",
    )

    assert first["status"] == "COMMITTED"
    assert first["episode_count"] == 2
    assert first["contains_eval_plane"] is False
    assert first["contains_gt"] is False
    assert first["contains_residual"] is False
    assert first_path.read_bytes() == second_path.read_bytes()
    assert first["sha256"] == second["sha256"]

    with zipfile.ZipFile(first_path) as archive:
        assert archive.namelist() == [
            "MANIFEST.json",
            "episodes/ep-000001.json",
            "episodes/ep-000002.json",
            "montages/ep-000001.png",
            "montages/ep-000002.png",
            "protocol/petct_codex_prompt_v2.json",
        ]
        manifest = json.loads(archive.read("MANIFEST.json"))
        assert manifest["episode_count"] == 2
        assert manifest["contains_eval_plane"] is False
        serialized = b"".join(
            archive.read(name)
            for name in archive.namelist()
            if name.startswith("episodes/")
        )
        lowered = serialized.lower()
        assert b"source_case_id" not in lowered
        assert b"source_patient_id" not in lowered
        assert b"gold_intent" not in lowered
        assert b"residual_sha256" not in lowered


def test_visible_packet_accepts_remove_background_scribble(tmp_path: Path) -> None:
    visible_root = tmp_path / "visible"
    montage_root = tmp_path / "montages"
    _write_episode(visible_root, "ep-000001", polarity="background")
    _write_montage(montage_root, "ep-000001")

    receipt = export_visible_packet(
        visible_root=visible_root,
        montage_root=montage_root,
        protocol_path=PROJECT / "protocols" / "petct_codex_prompt_v2.json",
        output_zip=tmp_path / "remove-visible-v2.zip",
        packet_id="remove-visible-v2",
    )

    assert receipt["contract_version"] == "PETCT-CODEX-VISIBLE-PACKET-v2.0"


@pytest.mark.parametrize(
    "forbidden_key",
    ["gold_intent", "residual_sha256", "source_case_id", "target_component"],
)
def test_export_rejects_evaluation_truth_in_visible_json(
    tmp_path: Path, forbidden_key: str
) -> None:
    visible_root = tmp_path / "visible"
    montage_root = tmp_path / "montages"
    _write_episode(visible_root, "ep-000001")
    _write_montage(montage_root, "ep-000001")
    path = visible_root / "ep-000001.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload[forbidden_key] = "leak"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(PacketContractError, match="forbidden evaluation field"):
        export_visible_packet(
            visible_root=visible_root,
            montage_root=montage_root,
            protocol_path=PROJECT / "protocols" / "petct_codex_prompt_v2.json",
            output_zip=tmp_path / "packet.zip",
            packet_id="pilot3-visible-v1",
        )


def test_export_rejects_missing_or_unreferenced_montage(tmp_path: Path) -> None:
    visible_root = tmp_path / "visible"
    montage_root = tmp_path / "montages"
    _write_episode(visible_root, "ep-000001")
    montage_root.mkdir(parents=True)

    with pytest.raises(PacketContractError, match="montage missing"):
        export_visible_packet(
            visible_root=visible_root,
            montage_root=montage_root,
            protocol_path=PROJECT / "protocols" / "petct_codex_prompt_v2.json",
            output_zip=tmp_path / "missing.zip",
            packet_id="pilot3-visible-v1",
        )

    _write_montage(montage_root, "ep-000001")
    _write_montage(montage_root, "ep-deadbeef")
    with pytest.raises(PacketContractError, match="unreferenced montage"):
        export_visible_packet(
            visible_root=visible_root,
            montage_root=montage_root,
            protocol_path=PROJECT / "protocols" / "petct_codex_prompt_v2.json",
            output_zip=tmp_path / "extra.zip",
            packet_id="pilot3-visible-v1",
        )


def test_export_is_no_clobber(tmp_path: Path) -> None:
    visible_root = tmp_path / "visible"
    montage_root = tmp_path / "montages"
    _write_episode(visible_root, "ep-000001")
    _write_montage(montage_root, "ep-000001")
    output = tmp_path / "pilot3-visible-v1.zip"
    export_visible_packet(
        visible_root=visible_root,
        montage_root=montage_root,
        protocol_path=PROJECT / "protocols" / "petct_codex_prompt_v2.json",
        output_zip=output,
        packet_id="pilot3-visible-v1",
    )

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        export_visible_packet(
            visible_root=visible_root,
            montage_root=montage_root,
            protocol_path=PROJECT / "protocols" / "petct_codex_prompt_v2.json",
            output_zip=output,
            packet_id="pilot3-visible-v1",
        )
