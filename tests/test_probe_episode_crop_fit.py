from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


PROJECT = Path(__file__).resolve().parents[1]
SCRIPTS = PROJECT / "scripts"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(SCRIPTS / "common"))
sys.path.insert(0, str(SCRIPTS / "data"))

import probe_episode_crop_fit as probe  # noqa: E402


CROP_ERROR_MARKER = "authorized COMPLETE/LOCAL target exceeds frozen crop"


def _row(episode_id: str, partition: str = "train") -> dict:
    return {
        "episode_id": episode_id,
        "partition": partition,
        "patient_id": "p1",
        "goal": "ADD_SAME_LOCAL",
        "coordinates_xyz": [[10, 11, 12]],
    }


def _manifest(tmp_path: Path, rows: list[dict]) -> Path:
    path = tmp_path / "episode_manifest.jsonl"
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    return path


def _config(tmp_path: Path) -> Path:
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(
            {
                "learning_tensor_normalization": {
                    "crop_field_mm": 192.0,
                    "output_size_px": 256,
                    "output_spacing_mm": 0.75,
                }
            }
        ),
        encoding="utf-8",
    )
    return path


def _run_probe(
    monkeypatch,
    tmp_path,
    rows,
    materialize,
    report,
    max_rows=None,
):
    manifest = _manifest(tmp_path, rows)
    monkeypatch.setattr(probe, "materialize_episode", materialize)
    monkeypatch.setattr(probe, "crop_offset_report", report)
    out = tmp_path / "kept.jsonl"
    excl = tmp_path / "exclusions.jsonl"
    code = probe.main(
        [
            "--episode-manifest", str(manifest),
            "--experiment-config", str(_config(tmp_path)),
            "--output-manifest", str(out),
            "--exclusions", str(excl),
            "--temp-root", str(tmp_path / "tmp"),
        ]
        + (["--max-rows", str(max_rows)] if max_rows is not None else [])
    )
    return code, out, excl


class TestClassifyRow:
    def test_keep(self, monkeypatch, tmp_path):
        calls = []

        def fake_materialize(row, **kwargs):
            calls.append(row["episode_id"])
            return {"ok": True}

        code, out, excl = _run_probe(
            monkeypatch,
            tmp_path,
            [_row("e1")],
            fake_materialize,
            lambda row, field_mm, expected_spacing: {"crop_limit_mm": 95.625},
        )
        assert code == 0
        assert calls == ["e1"]
        kept = [json.loads(line) for line in out.open()]
        assert [r["episode_id"] for r in kept] == ["e1"]
        assert excl.read_text() == ""

    def test_crop_excluded(self, monkeypatch, tmp_path):
        def fake_materialize(row, **kwargs):
            if row["episode_id"] == "e2":
                raise RuntimeError(CROP_ERROR_MARKER)
            return {"ok": True}

        code, out, excl = _run_probe(
            monkeypatch,
            tmp_path,
            [_row("e1"), _row("e2")],
            fake_materialize,
            lambda row, field_mm, expected_spacing: {
                "crop_limit_mm": 95.625,
                "max_authorized_offset_mm": 120.0,
                "overshoot_mm": 24.375,
            },
        )
        assert code == 0
        kept = [json.loads(line) for line in out.open()]
        assert [r["episode_id"] for r in kept] == ["e1"]
        excluded = [json.loads(line) for line in excl.open()]
        assert len(excluded) == 1
        record = excluded[0]
        assert record["row"]["episode_id"] == "e2"
        assert record["error"] == CROP_ERROR_MARKER
        assert record["report"]["overshoot_mm"] == 24.375

    def test_other_error_aborts_without_outputs(self, monkeypatch, tmp_path):
        def fake_materialize(row, **kwargs):
            raise ValueError("unexpected defect")

        code, kept_path, excl_path = _run_probe(
            monkeypatch,
            tmp_path,
            [_row("e1")],
            fake_materialize,
            lambda row, field_mm, expected_spacing: {},
        )
        assert code == 1
        assert not kept_path.exists()
        assert not excl_path.exists()

    def test_partition_test_aborts(self, monkeypatch, tmp_path):
        with pytest.raises(SystemExit):
            _run_probe(
                monkeypatch,
                tmp_path,
                [_row("e1", partition="test")],
                lambda row, **kwargs: {"ok": True},
                lambda row, field_mm, expected_spacing: {},
            )

    def test_max_rows_limits(self, monkeypatch, tmp_path):
        code, out, _ = _run_probe(
            monkeypatch,
            tmp_path,
            [_row("e1"), _row("e2"), _row("e3")],
            lambda row, **kwargs: {"ok": True},
            lambda row, field_mm, expected_spacing: {},
            max_rows=2,
        )
        assert code == 0
        kept = [json.loads(line) for line in out.open()]
        assert [r["episode_id"] for r in kept] == ["e1", "e2"]

    def test_materialize_failure_leaves_no_staged_files(self, monkeypatch, tmp_path):
        seen = []

        def fake_materialize(row, **kwargs):
            staged = Path(kwargs["staged_visible_root"]) / (
                row["episode_id"] + ".npz"
            )
            staged.write_bytes(b"x")
            seen.append(staged)
            raise RuntimeError(CROP_ERROR_MARKER)

        code, _, _ = _run_probe(
            monkeypatch,
            tmp_path,
            [_row("e1")],
            fake_materialize,
            lambda row, field_mm, expected_spacing: {},
        )
        assert code == 0
        for staged in seen:
            assert not staged.exists()
