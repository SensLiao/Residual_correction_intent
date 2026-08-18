"""Contract tests for the R13-trajectory-5r orchestration launcher.

The launcher runs a pure-CPU chain (preflight -> lineage -> trajectory build
-> tensors -> candidates -> three-lane manifests -> pointer targets -> seal)
that consumes the frozen R13-main residuals and never touches a GPU or the
locked test partition.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

PROJECT = Path(__file__).resolve().parents[1]


def _usable_bash() -> str | None:
    candidates = [r"C:\Program Files\Git\bin\bash.exe", shutil.which("bash")]
    for candidate in candidates:
        if not candidate:
            continue
        try:
            probe = subprocess.run(
                [candidate, "--version"],
                capture_output=True,
                text=True,
                encoding="utf-8",
            )
        except OSError:
            continue
        if probe.returncode == 0 and "bash" in probe.stdout.lower():
            return candidate
    return None


BASH = _usable_bash()
requires_bash = pytest.mark.skipif(BASH is None, reason="bash is unavailable")


def _launcher() -> Path:
    return (
        PROJECT
        / "scripts"
        / "orchestration"
        / "launch_petct_r13_trajectory_5r.sh"
    )


def _posix(path: Path) -> str:
    text = str(path).replace("\\", "/")
    if len(text) >= 2 and text[1] == ":":
        text = "/" + text[0].lower() + text[2:]
    return text


def _run_root() -> str:
    return _posix(PROJECT / "route_a" / "runs") + "/PETCT-R13-TRAJECTORY-5R-DRYTEST"


def _run_launcher(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [BASH, str(_launcher()), *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
    )


def _dry_run_args() -> list[str]:
    base = _posix(PROJECT)
    return [
        "--run-root",
        _run_root(),
        "--r13-main-data",
        base + "/route_a/runs/PETCT-R13-MAIN-DRYTEST/R13-main",
        "--r13-main-data-ready",
        base + "/route_a/runs/PETCT-R13-MAIN-DRYTEST/R13-main/data-ready.json",
        "--oof-ready",
        base + "/records/oof-ready.json",
        "--learning-split",
        base + "/records/learning-split.json",
        "--experiment-config",
        base + "/configs/petct_route_a_experiment.json",
        "--official-simulator",
        base + "/external_runners/autopetv_protocol/interactive/simulate_scribbles.py",
        "--official-runtime-manifest",
        base + "/protocols/autopetv_protocol_runtime.json",
        "--dry-run",
    ]


def test_launcher_wires_the_five_round_cpu_chain() -> None:
    text = _launcher().read_text(encoding="utf-8")
    assert "build_petct_r13_trajectory_5r.py" in text
    assert "materialize_petct_r13_trajectory_5r_tensors.py" in text
    assert "materialize_petct_r13_trajectory_5r_programs.py" in text
    assert "petct_trajectory_lineage.py" in text
    # The frozen-input preflight validates the active R13-main corpus receipt.
    assert "validate-data" in text
    assert "--r13-main-data-ready" in text
    # R13-main residuals are consumed, never regenerated.
    assert "audit-only/residuals.jsonl" in text
    assert "audit-only/RESIDUAL_READY.json" in text
    assert "build_petct_residual_manifest.py" not in text
    # Parity-bound single-round generation contract.
    assert "--strategy-mode" in text and "primary" in text
    assert "--seed" in text and "42" in text
    assert "--partitions" in text and "train val" in text
    # Independent dataset identity and fail-closed state machine.
    assert "R13-trajectory-5r" in text
    assert "PETCT-R13-TRAJECTORY-5R-" in text
    assert "M0_V6_FIVEFOLD_OOF" in text
    assert "TMPDIR" in text
    assert "/mnt/HDD4/zlei0805/honor_degree/.tmp" in text
    assert "--dry-run" in text
    # Candidate/pointer materializers are reused unchanged (no 5r fork).
    assert "materialize_petct_component_candidates.py" in text
    assert "materialize_petct_component_targets.py" in text


@requires_bash
def test_launcher_dry_run_emits_expected_plan() -> None:
    result = _run_launcher(*_dry_run_args())
    assert result.returncode == 0, result.stderr
    steps = [json.loads(line) for line in result.stdout.splitlines() if line.strip()]
    names = [step["step"] for step in steps]
    assert names == [
        "preflight",
        "lineage",
        "trajectories",
        "tensors",
        "candidates",
        "three-manifests",
        "pointer-targets",
        "seal",
    ]
    by_name = {step["step"]: step for step in steps}
    preflight = " ".join(by_name["preflight"]["commands"])
    assert "validate-data" in preflight
    assert "data-ready.json" in preflight
    lineage = " ".join(by_name["lineage"]["commands"])
    assert "petct_trajectory_lineage.py" in lineage
    assert "issue" in lineage
    trajectories = " ".join(by_name["trajectories"]["commands"])
    assert "build_petct_r13_trajectory_5r.py" in trajectories
    assert "residuals.jsonl" in trajectories
    assert "RESIDUAL_READY.json" in trajectories
    assert "trajectory-states" in trajectories
    assert "trajectories.jsonl" in trajectories
    tensors = " ".join(by_name["tensors"]["commands"])
    assert "materialize_petct_r13_trajectory_5r_tensors.py" in tensors
    candidates = " ".join(by_name["candidates"]["commands"])
    assert "materialize_petct_component_candidates.py" in candidates
    manifests = " ".join(by_name["three-manifests"]["commands"])
    assert "materialize_petct_r13_trajectory_5r_programs.py" in manifests
    targets = " ".join(by_name["pointer-targets"]["commands"])
    assert "materialize_petct_component_targets.py" in targets
    seal = " ".join(by_name["seal"]["commands"])
    assert "petct_trajectory_lineage.py" in seal
    assert "seal" in seal
    assert "trajectory-data-ready.json" in seal


@requires_bash
def test_launcher_rejects_bad_arguments_fail_closed() -> None:
    args = _dry_run_args()
    # Unknown flag.
    assert _run_launcher(*args, "--bogus").returncode == 2
    # Real mode without --dry-run must fail closed (inputs do not exist).
    assert _run_launcher(*[a for a in args if a != "--dry-run"]).returncode == 2
    # Run root outside the R13-trajectory-5r naming pattern.
    bad = list(args)
    bad[bad.index("--run-root") + 1] = "/tmp/not-a-run-root"
    assert _run_launcher(*bad).returncode == 2
    # Missing required argument.
    assert _run_launcher("--run-root", _run_root()).returncode == 2
