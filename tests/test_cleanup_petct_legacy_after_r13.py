from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


PROJECT = Path(__file__).resolve().parents[1]


def _module():
    path = PROJECT / "scripts" / "orchestration" / "cleanup_petct_legacy_after_r13.py"
    assert path.is_file(), "R13 cleanup gate is missing"
    spec = importlib.util.spec_from_file_location("r13_cleanup", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cleanup_plan_resolves_only_explicit_legacy_targets(tmp_path: Path) -> None:
    module = _module()
    project = tmp_path / "project"
    target = project / "route_a" / "runs" / "legacy"
    target.mkdir(parents=True)
    plan = {
        "delete_after_r13_gates": [
            {"path": "route_a/runs/legacy", "size": "1G", "reason": "fixture"}
        ],
        "delete_gate": {"forbidden_targets": ["data", "records", "upstream"]},
    }
    resolved = module.resolve_cleanup_targets(project, plan)
    assert resolved == [target.resolve()]

    plan["delete_after_r13_gates"][0]["path"] = "data"
    (project / "data").mkdir()
    with pytest.raises(module.CleanupContractError, match="forbidden"):
        module.resolve_cleanup_targets(project, plan)


def test_cleanup_refuses_before_r13_gates(tmp_path: Path) -> None:
    module = _module()
    with pytest.raises(module.CleanupContractError, match="data-ready"):
        module.validate_cleanup_gates(
            tmp_path / "missing-r13", tmp_path / "missing-smoke"
        )
