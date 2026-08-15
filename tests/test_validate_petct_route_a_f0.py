from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, Sequence, Union

import pytest


PROJECT = Path(__file__).resolve().parents[1]
ORCHESTRATION = PROJECT / "scripts" / "orchestration"
sys.path.insert(0, str(ORCHESTRATION))

from validate_petct_route_a_f0 import (  # noqa: E402
    BLOCKER_IDS,
    F0_TEST_FILES,
    REQUIRED_SOURCE_FILES,
    F0Error,
    _canonical_sha256,
    issue_receipt,
    validate_receipt,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(text)
    return path


def _copy_bytes(source: Path, target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(source.read_bytes())
    return target


def _fixture(tmp_path: Path) -> Dict[str, Union[Path, str]]:
    project = tmp_path / "project"
    project.mkdir()
    for relative in REQUIRED_SOURCE_FILES:
        _write(project / relative, f"# {relative}\n")
    for relative in F0_TEST_FILES:
        _write(project / relative, f"def test_{Path(relative).stem}():\n    pass\n")

    config = _write(project / "configs" / "petct_route_a_experiment.json", "{}\n")
    simulator = _copy_bytes(
        PROJECT / "upstream" / "autoPETV" / "interactive" / "simulate_scribbles.py",
        project / "external_runners" / "autopetv_protocol" / "interactive" / "simulate_scribbles.py",
    )
    metrics = _copy_bytes(
        PROJECT / "upstream" / "autoPETV" / "metrics.py",
        project / "external_runners" / "autopetv_protocol" / "metrics.py",
    )
    _copy_bytes(
        PROJECT / "upstream" / "autoPETV" / "LICENSE",
        project / "external_runners" / "autopetv_protocol" / "LICENSE",
    )
    runtime_manifest = _copy_bytes(
        PROJECT / "protocols" / "autopetv_protocol_runtime.json",
        project / "protocols" / "autopetv_protocol_runtime.json",
    )

    core_receipt_bytes = (
        json.dumps(
            {
                "schema_version": "PETCT-NNUNET-ENV-v1.1",
                "status": "PREFLIGHT_PASS_PENDING_ATOMIC_EVIDENCE_PUBLICATION",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    receipt_sha = hashlib.sha256(core_receipt_bytes.encode("utf-8")).hexdigest()
    core = {
        "schema_version": "PETCT-NNUNET-ENV-EVIDENCE-BUNDLE-v1.0",
        "status": "ENVIRONMENT_EVIDENCE_COMPLETE",
        "setup_mode": "VERIFY_EXISTING_NO_INSTALL",
        "files": [
            {
                "name": "petct_nnunet_v281.json",
                "sha256": receipt_sha,
                "bytes": len(core_receipt_bytes.encode("utf-8")),
            }
        ],
    }
    bundle_sha = _canonical_sha256(core)
    bundle_root = project / "nnunet" / "envs" / "evidence-bundles" / bundle_sha
    core_receipt = _write(
        bundle_root / "petct_nnunet_v281.json", core_receipt_bytes
    )
    bundle_manifest = _write(
        bundle_root / "bundle.json",
        json.dumps(
            {**core, "bundle_sha256": bundle_sha}, indent=2, sort_keys=True
        )
        + "\n",
    )
    marker = _write(
        project / "nnunet" / "envs" / "ENV_READY.done",
        json.dumps(
            {
                "schema_version": "PETCT-NNUNET-ENV-MARKER-v1.0",
                "status": "ENVIRONMENT_EVIDENCE_COMPLETE",
                "setup_mode": "VERIFY_EXISTING_NO_INSTALL",
                "bundle_path": str(bundle_root.resolve()),
                "bundle_sha256": bundle_sha,
                "bundle_manifest_sha256": _sha(bundle_manifest),
                "receipt_path": str(core_receipt.resolve()),
                "receipt_sha256": _sha(core_receipt),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )
    return {
        "project": project,
        "config": config,
        "simulator": simulator,
        "metrics": metrics,
        "runtime_manifest": runtime_manifest,
        "marker": marker,
        "bundle_sha": bundle_sha,
        "output": project / "route_a" / "manifests" / "F0_READY.json",
        "log": project / "route_a" / "manifests" / "F0_TESTS.log",
    }


def _pass_runner(
    command: Sequence[str], project: Path
) -> subprocess.CompletedProcess[str]:
    assert list(command)[1:5] == ["-m", "pytest", "-q", "--disable-warnings"]
    assert list(command)[5:] == list(F0_TEST_FILES)
    assert project.is_dir()
    return subprocess.CompletedProcess(list(command), 0, stdout="137 passed in 1.00s\n")


def _issue(fixture: Dict[str, Union[Path, str]], **overrides):
    arguments = {
        "project_root": fixture["project"],
        "experiment_config": fixture["config"],
        "environment_marker": fixture["marker"],
        "official_simulator": fixture["simulator"],
        "official_metrics": fixture["metrics"],
        "official_runtime_manifest": fixture["runtime_manifest"],
        "output": fixture["output"],
        "test_log": fixture["log"],
        "expected_env_bundle_sha256": fixture["bundle_sha"],
        "test_runner": _pass_runner,
    }
    arguments.update(overrides)
    return issue_receipt(**arguments)


def _validate(fixture: Dict[str, Union[Path, str]], **overrides):
    arguments = {
        "receipt_path": fixture["output"],
        "project_root": fixture["project"],
        "experiment_config": fixture["config"],
        "environment_marker": fixture["marker"],
        "official_simulator": fixture["simulator"],
        "official_metrics": fixture["metrics"],
        "official_runtime_manifest": fixture["runtime_manifest"],
        "expected_env_bundle_sha256": fixture["bundle_sha"],
    }
    arguments.update(overrides)
    return validate_receipt(**arguments)


def test_issue_runs_fixed_suite_and_revalidates_exact_seven_gates(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    payload = _issue(fixture)
    summary = _validate(fixture)

    assert payload["closed_blocker_ids"] == list(BLOCKER_IDS)
    assert len(payload["closed_blocker_ids"]) == 7
    assert payload["test_evidence"]["command_argv"][5:] == list(F0_TEST_FILES)
    assert payload["test_evidence"]["exit_code"] == 0
    assert payload["environment"]["bundle_sha256"] == fixture["bundle_sha"]
    assert payload["scientific_execution"] is False
    assert payload["scientific_result_claim"] is False
    assert summary["receipt"]["sha256"] == _sha(fixture["output"])
    assert Path(fixture["log"]).read_text(encoding="utf-8").startswith("argv=")


def test_issue_is_no_clobber_and_never_publishes_a_failed_suite(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)

    def fail_runner(command, project):
        return subprocess.CompletedProcess(list(command), 1, stdout="1 failed\n")

    with pytest.raises(F0Error, match="fixed F0 pytest suite failed"):
        _issue(fixture, test_runner=fail_runner)
    assert not Path(fixture["output"]).exists()
    assert not Path(fixture["log"]).exists()

    _issue(fixture)
    with pytest.raises(F0Error, match="refusing overwrite"):
        _issue(fixture)


def test_validation_rejects_source_config_and_log_drift(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    _issue(fixture)

    source = Path(fixture["project"]) / REQUIRED_SOURCE_FILES[0]
    source.write_text("# tampered\n", encoding="utf-8")
    with pytest.raises(F0Error, match="source bundle differs"):
        _validate(fixture)
    _write(source, f"# {REQUIRED_SOURCE_FILES[0]}\n")

    _write(Path(fixture["config"]), '{"changed": true}\n')
    with pytest.raises(F0Error, match="canonical config"):
        _validate(fixture)
    _write(Path(fixture["config"]), "{}\n")

    Path(fixture["log"]).write_text("argv=tampered\n", encoding="utf-8")
    with pytest.raises(F0Error, match="log hash/size mismatch"):
        _validate(fixture)


def test_validation_rejects_inexact_blocker_set_and_environment_bundle(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    _issue(fixture)
    receipt_path = Path(fixture["output"])
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["closed_blocker_ids"] = list(BLOCKER_IDS[:-1])
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with pytest.raises(F0Error, match="exact seven blocker"):
        _validate(fixture)

    receipt["closed_blocker_ids"] = list(BLOCKER_IDS)
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with pytest.raises(F0Error, match="expected immutable bundle"):
        _validate(fixture, expected_env_bundle_sha256="0" * 64)


def test_validation_rejects_receipt_symlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path)
    _issue(fixture)
    receipt = Path(fixture["output"]).absolute()
    original = Path.is_symlink

    def fake_is_symlink(path: Path) -> bool:
        return path.absolute() == receipt or original(path)

    monkeypatch.setattr(Path, "is_symlink", fake_is_symlink)
    with pytest.raises(F0Error, match="symlink"):
        _validate(fixture)


def test_issue_rejects_source_drift_during_fixed_tests(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    source = Path(fixture["project"]) / REQUIRED_SOURCE_FILES[0]

    def mutating_runner(command, project):
        source.write_text("# changed during pytest\n", encoding="utf-8")
        return subprocess.CompletedProcess(list(command), 0, stdout="all passed\n")

    with pytest.raises(F0Error, match="changed while the fixed suite was running"):
        _issue(fixture, test_runner=mutating_runner)
    assert not Path(fixture["output"]).exists()
    assert not Path(fixture["log"]).exists()


def test_f0_rejects_missing_or_drifted_minimal_runtime_before_publication(
    tmp_path: Path,
) -> None:
    missing_root = tmp_path / "missing"
    missing_root.mkdir()
    missing = _fixture(missing_root)
    simulator = Path(missing["simulator"])
    license_path = simulator.parent.parent / "LICENSE"
    license_path.unlink()
    with pytest.raises(F0Error, match="LICENSE"):
        _issue(missing)
    assert not Path(missing["output"]).exists()

    drifted_root = tmp_path / "drifted"
    drifted_root.mkdir()
    drifted = _fixture(drifted_root)
    drifted_license = Path(drifted["simulator"]).parent.parent / "LICENSE"
    drifted_license.write_text("drifted\n", encoding="utf-8")
    with pytest.raises(F0Error, match="hash mismatch: LICENSE"):
        _issue(drifted)
    assert not Path(drifted["output"]).exists()
