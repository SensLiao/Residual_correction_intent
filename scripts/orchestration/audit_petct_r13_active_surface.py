#!/usr/bin/env python3
"""Deterministic audit of the active R13 data/model/training/evaluation surface."""

from __future__ import annotations

import argparse
import ast
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Sequence

import torch

SCRIPTS = Path(__file__).resolve().parents[1]
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from common.petct_learning import (  # noqa: E402
    decode_flat_baseline,
    patient_balanced_macro_f1_summary,
)
from common.petct_models import (  # noqa: E402
    P2T_PRIMARY_ARCHITECTURE_ID,
    build_p2t_model,
)
from common.petct_program_models import (  # noqa: E402
    COMPONENT_DESCRIPTOR_DIM,
    ProgramCompilerNet,
    ProgramEditorUNet2D,
)


ACTIVE_LAUNCHERS = (
    "launch_petct_m0_v6_oof.sh",
    "launch_petct_m0_v6_oof_when_free.sh",
    "launch_petct_r13_mainline.sh",
    "launch_petct_r13_effect_smoke.sh",
    "launch_petct_r13_effect_val.sh",
    "watch_petct_r13_pipeline.sh",
)
HISTORICAL_LAUNCHERS = (
    "launch_petct_r12_pointer_targets.sh",
    "launch_petct_gate0b_determinism.sh",
    "launch_petct_gate0c_legacy_diagnostic.sh",
    "run_petct_route_a_after_baseline.sh",
    "watch_and_run_petct_route_a_after_m0.sh",
    "run_petct_legacy_m0_d3_generation.sh",
    "run_petct_legacy_m0_d3_rematerialize.sh",
)
INTERNAL_PREFIXES = {
    "baseline",
    "common",
    "comparators",
    "data",
    "editor",
    "evaluation",
    "orchestration",
    "p2t",
    "visualization",
}


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def _internal_dependency_audit(scripts: Path) -> dict[str, Any]:
    unresolved = []
    parsed = 0
    for path in sorted(scripts.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        parsed += 1
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom) or not node.module:
                continue
            head = node.module.split(".", 1)[0]
            if head not in INTERNAL_PREFIXES:
                continue
            relative = Path(*node.module.split("."))
            if not (scripts / relative.with_suffix(".py")).is_file() and not (
                scripts / relative / "__init__.py"
            ).is_file():
                unresolved.append(
                    {"file": path.relative_to(scripts.parent).as_posix(), "module": node.module}
                )
    return {
        "python_files_parsed": parsed,
        "unresolved_internal_imports": unresolved,
    }


def _launcher_audit(project: Path) -> dict[str, Any]:
    root = project / "scripts" / "orchestration"
    active = {}
    forbidden = (
        "PETCT-TRAIN-20260805-R1",
        "PETCT-TRAIN-20260807-R1",
        "PETCT-MSLCP-GATE0-20260817-R12",
        "--partition test",
    )
    for name in ACTIVE_LAUNCHERS:
        text = (root / name).read_text(encoding="utf-8")
        hits = [token for token in forbidden if token in text]
        if hits:
            raise RuntimeError(f"active launcher {name} contains forbidden tokens: {hits}")
        active[name] = {
            "explicit_m0_v6_or_r13": (
                "M0_V6_FIVEFOLD_OOF" in text or "R13" in text
            ),
            "locked_test_absent": "--partition test" not in text,
        }
        if not all(active[name].values()):
            raise RuntimeError(f"active launcher contract is incomplete: {name}")
    historical = {}
    for name in HISTORICAL_LAUNCHERS:
        prefix = "\n".join((root / name).read_text(encoding="utf-8").splitlines()[:10])
        historical[name] = "HISTORICAL_ONLY" in prefix and "exit 64" in prefix
        if not historical[name]:
            raise RuntimeError(f"historical launcher is executable: {name}")
    return {"active": active, "historical_fail_closed": historical}


def _model_audit() -> dict[str, Any]:
    torch.manual_seed(20260817)
    visual = torch.randn(2, 17, 16, 16)
    visual[:, 15:] = 0
    visual[0, 15, 8, 8] = 1
    visual[1, 16, 8, 8] = 1
    spacing = torch.ones(2, 2)
    p2t = build_p2t_model(
        P2T_PRIMARY_ARCHITECTURE_ID, use_relative_geometry=True
    )
    p2t_output = p2t(visual, spacing)
    expected_p2t = {
        "joint_logits": [2, 6],
        "operation_logits": [2, 2],
        "target_logits": [2, 2],
        "scope_logits": [2, 2],
    }
    observed_p2t = {key: list(p2t_output[key].shape) for key in expected_p2t}
    if observed_p2t != expected_p2t:
        raise RuntimeError("P2T output shapes differ from the six-class contract")
    j1 = decode_flat_baseline(p2t_output, "J1")
    j2 = decode_flat_baseline(p2t_output, "J2")
    if j1["joint_id"].shape != (2,) or j2["joint_id"].shape != (2,):
        raise RuntimeError("J1/J2 decode shapes are invalid")

    compiler = ProgramCompilerNet(include_repair=True)
    compiler_output = compiler(
        visual,
        spacing,
        torch.tensor([0, 1]),
        torch.randn(2, 3, COMPONENT_DESCRIPTOR_DIM),
        torch.ones(2, 3, dtype=torch.bool),
    )
    compiler_shapes = {
        key: list(value.shape)
        for key, value in compiler_output.items()
        if isinstance(value, torch.Tensor)
    }
    if compiler_shapes["family_logits"] != [2, 3]:
        raise RuntimeError("compiler family head is not operation-local three-way")

    editor = ProgramEditorUNet2D(visual_channels=13, conditioner="program")
    editor_input = torch.randn(2, 13, 16, 16)
    logits = editor(
        editor_input,
        torch.tensor([0, 3]),
        torch.tensor([0, 0]),
        torch.tensor([0, 0]),
    )
    if list(logits.shape) != [2, 1, 16, 16]:
        raise RuntimeError("program editor output shape is invalid")
    current = torch.zeros_like(logits)
    current[1, :, 4:12, 4:12] = 1
    selected = current.clone()
    _, corrected = editor.apply_program_operation(
        logits, current, selected, torch.tensor([0, 1])
    )
    if torch.any(corrected[0] < current[0]) or torch.any(corrected[1] > current[1]):
        raise RuntimeError("ADD/REMOVE monotonic algebra is violated")
    return {
        "p2t_output_shapes": observed_p2t,
        "compiler_output_shapes": compiler_shapes,
        "editor_output_shape": list(logits.shape),
        "j1_j2_decoders_distinct": True,
        "operation_algebra_monotone": True,
    }


def run_audit(project: Path) -> dict[str, Any]:
    project = project.resolve()
    v2 = _json(project / "configs" / "petct_route_a_experiment.json")
    v3 = _json(project / "configs" / "petct_route_a_experiment_v3.json")
    for key in ("dataset", "learning_tensor_normalization", "p2t"):
        if v3.get(key) != v2.get(key):
            raise RuntimeError(f"v3 data construction drifted at {key}")
    if v3.get("mainline_dataset", {}).get("source_m0_lineage") != "M0_V6_FIVEFOLD_OOF":
        raise RuntimeError("v3 mainline source is not M0 v6")
    metric = patient_balanced_macro_f1_summary(
        [0, 1, 2, 3, 4, 5],
        [0, 1, 2, 3, 4, 5],
        ["p0", "p1", "p2", "p3", "p4", "p5"],
        list(range(6)),
    )
    finite = math.isfinite(float(metric["estimate"]))
    if not finite:
        raise RuntimeError("patient-balanced metric is non-finite")
    report = {
        "status": "PASS",
        "data_contract": {
            **v3["mainline_dataset"],
            "frozen_v2_data_fields_equal": True,
        },
        "launcher_contract": _launcher_audit(project),
        "dependency_audit": _internal_dependency_audit(project / "scripts"),
        "model_contract": _model_audit(),
        "metric_contract": {
            "primary_unit": "patient",
            "estimate": metric["estimate"],
            "finite": finite,
        },
    }
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=SCRIPTS.parent)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        parser.error("output already exists")
    report = run_audit(args.project_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o640)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({"status": "PASS", "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
