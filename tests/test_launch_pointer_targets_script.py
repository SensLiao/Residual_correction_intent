from __future__ import annotations

from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]


def test_r12_pointer_target_launcher_is_retired_fail_closed() -> None:
    launcher = (
        PROJECT / "scripts" / "orchestration" / "launch_petct_r12_pointer_targets.sh"
    )
    text = launcher.read_text(encoding="utf-8")

    assert "HISTORICAL_ONLY" in text
    assert "launch_petct_r13_mainline.sh" in text
    assert "exit 64" in text
    assert "materialize_petct_component_targets.py" not in text
