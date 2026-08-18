from __future__ import annotations

from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]


def test_gate0c_launcher_is_retired_fail_closed() -> None:
    launcher = (
        PROJECT / "scripts" / "orchestration" / "launch_petct_gate0c_legacy_diagnostic.sh"
    )
    text = launcher.read_text(encoding="utf-8")

    assert "HISTORICAL_ONLY" in text
    assert "exit 64" in text
    assert "infer_petct_residual_editor.py" not in text
