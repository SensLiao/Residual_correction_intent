from __future__ import annotations

import importlib.util
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]


def test_active_surface_audit_closes_dependencies_models_io_and_metrics() -> None:
    path = PROJECT / "scripts" / "orchestration" / "audit_petct_r13_active_surface.py"
    assert path.is_file(), "R13 active-surface audit is missing"
    spec = importlib.util.spec_from_file_location("r13_audit", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    report = module.run_audit(PROJECT)
    assert report["status"] == "PASS"
    assert report["data_contract"]["source_m0_lineage"] == "M0_V6_FIVEFOLD_OOF"
    assert report["dependency_audit"]["unresolved_internal_imports"] == []
    assert report["model_contract"]["p2t_output_shapes"]["joint_logits"] == [2, 6]
    assert report["model_contract"]["compiler_output_shapes"]["family_logits"] == [2, 3]
    assert report["model_contract"]["editor_output_shape"] == [2, 1, 16, 16]
    assert report["metric_contract"]["finite"] is True
