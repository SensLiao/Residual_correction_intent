from __future__ import annotations

from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]


def test_r12_gate0b_launcher_is_retired_fail_closed() -> None:

    launcher = (
        PROJECT / "scripts" / "orchestration" / "launch_petct_gate0b_determinism.sh"
    )
    text = launcher.read_text(encoding="utf-8")

    assert "HISTORICAL_ONLY" in text
    assert "exit 64" in text
    assert "train_petct_program_v3.py" not in text
