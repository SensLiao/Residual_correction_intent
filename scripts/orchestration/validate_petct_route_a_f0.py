#!/usr/bin/env python3
"""Issue and validate the pre-run Route A F0 readiness receipt.

This is a leaf gate, not an experiment launcher.  ``issue`` runs the fixed
contract-test set without touching scientific data or a GPU, then binds the
deployed source tree, the canonical inputs, and the immutable core-environment
bundle.  ``validate`` recomputes every binding.  Both commands are standard-
library only so the gate can fail closed before a Route A run root is created.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence


SCHEMA_VERSION = "PETCT-ROUTE-A-F0-READY-v1.0"
EXPECTED_ENV_BUNDLE_SHA256 = (
    "87a2261af9d99eb8232a078a2f7ba81cf9f3b4a6389410c296ca9b8671246006"
)
EXPECTED_AUTOPETV_RUNTIME_MANIFEST_SHA256 = (
    "100e4e6453dbf88d3aebc8b4c8107f574c35c08119a1d6789f9fd166d13c4854"
)
AUTOPETV_RUNTIME_SCHEMA = "PETCT-AUTOPETV-PROTOCOL-RUNTIME-v2.0"
AUTOPETV_RUNTIME_STATUS = (
    "FROZEN_MINIMAL_RUNTIME_SIX_CLASS_POLARITY_ADAPTER_NOT_EXECUTED"
)
AUTOPETV_RUNTIME_COMMIT = "4a2026866bfacc812492cfc7e6a8c54ac3c4f703"
AUTOPETV_RUNTIME_FILES = (
    "LICENSE",
    "interactive/simulate_scribbles.py",
    "metrics.py",
)
BLOCKER_IDS = (
    "PRE-OOF-01",
    "PRE-DATA-02",
    "PRE-P2T-03",
    "PRE-ACCESS-04",
    "PRE-TRUTH-05",
    "PRE-SCRIBBLE-06",
    "PRE-COHORT-07",
)

# These files are the direct production surfaces for the seven blockers and
# the pre-run enforcement point.  The receipt additionally hashes every .py
# and .sh file below scripts/, so transitive code drift also invalidates F0.
REQUIRED_SOURCE_FILES = (
    "scripts/baseline/validate_petct_m0_preprocess.py",
    "scripts/baseline/validate_petct_m0_oof.py",
    "scripts/data/build_petct_residual_manifest.py",
    "scripts/data/build_petct_source_case_manifest.py",
    "scripts/data/build_petct_scribble_dataset.py",
    "scripts/data/build_petct_scribble_episode.py",
    "scripts/evaluation/evaluate_petct_m0_oof.py",
    "scripts/evaluation/evaluate_petct_p2t.py",
    "scripts/orchestration/run_petct_route_a_after_baseline.sh",
    "scripts/orchestration/validate_petct_route_a_f0.py",
    "scripts/orchestration/validate_petct_route_a_receipt_pipeline.py",
    "scripts/orchestration/watch_and_run_petct_route_a_after_m0.sh",
    "protocols/autopetv_protocol_runtime.json",
)

# ``issue`` owns this argv.  Callers cannot replace it with a smaller suite.
# The validator's own unit tests run separately; including them here would make
# ``issue`` recursively issue another receipt.
F0_TEST_FILES = (
    "tests/test_validate_petct_m0_preprocess.py",
    "tests/test_validate_petct_m0_oof.py",
    "tests/test_build_petct_residual_manifest.py",
    "tests/test_build_petct_source_case_manifest.py",
    "tests/test_build_petct_matched_state_dataset.py",
    "tests/test_build_petct_scribble_episode.py",
    "tests/test_build_petct_scribble_dataset.py",
    "tests/test_evaluate_petct_m0_oof.py",
    "tests/test_evaluate_petct_p2t.py",
    "tests/test_validate_petct_route_a_receipt_pipeline.py",
    "tests/test_run_petct_route_a_after_baseline.py",
    "tests/test_watch_and_run_petct_route_a_after_m0.py",
)


class F0Error(RuntimeError):
    """Raised when F0 cannot be truthfully issued or revalidated."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _reject_nonfinite(value: Any, *, label: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise F0Error(f"{label} contains a non-finite number")
    if isinstance(value, Mapping):
        for key, nested in value.items():
            _reject_nonfinite(nested, label=f"{label}.{key}")
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            _reject_nonfinite(nested, label=f"{label}[{index}]")


def _project_root(path: Path) -> Path:
    if path.is_symlink() or not path.is_dir():
        raise F0Error(f"project root must be a non-symlink directory: {path}")
    return path.resolve()


def _reject_symlink_chain(project: Path, path: Path, *, label: str) -> None:
    try:
        relative = path.absolute().relative_to(project.absolute())
    except ValueError as error:
        raise F0Error(f"{label} is outside the project root: {path}") from error
    cursor = project
    for part in relative.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise F0Error(f"{label} traverses a symlink: {cursor}")


def _regular(project: Path, path: Path, *, label: str) -> Path:
    candidate = path if path.is_absolute() else project / path
    _reject_symlink_chain(project, candidate, label=label)
    if candidate.is_symlink() or not candidate.is_file():
        raise F0Error(f"{label} must be a regular non-symlink file: {candidate}")
    resolved = candidate.resolve()
    if not resolved.is_relative_to(project):
        raise F0Error(f"{label} resolves outside the project root: {candidate}")
    return resolved


def _directory(project: Path, path: Path, *, label: str) -> Path:
    candidate = path if path.is_absolute() else project / path
    _reject_symlink_chain(project, candidate, label=label)
    if candidate.is_symlink() or not candidate.is_dir():
        raise F0Error(f"{label} must be a non-symlink directory: {candidate}")
    resolved = candidate.resolve()
    if not resolved.is_relative_to(project):
        raise F0Error(f"{label} resolves outside the project root: {candidate}")
    return resolved


def _absolute_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
    }


def _relative_record(project: Path, path: Path) -> dict[str, Any]:
    return {
        "path": path.resolve().relative_to(project).as_posix(),
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
    }


def _source_snapshot(project: Path) -> dict[str, Any]:
    scripts = _directory(project, project / "scripts", label="scripts root")
    for relative in REQUIRED_SOURCE_FILES:
        _regular(project, project / relative, label=f"required source {relative}")
    files: list[Path] = []
    for candidate in sorted(scripts.rglob("*"), key=lambda item: item.as_posix()):
        if candidate.is_symlink():
            raise F0Error(f"scripts source tree contains a symlink: {candidate}")
        if candidate.is_file() and candidate.suffix in {".py", ".sh"}:
            if "__pycache__" not in candidate.parts:
                files.append(candidate.resolve())
    if not files:
        raise F0Error("scripts source bundle is empty")
    records = [_relative_record(project, path) for path in files]
    canonical_lines = "".join(
        f'{record["sha256"]}  {record["path"]}\n' for record in records
    )
    return {
        "algorithm": "sha256-lines-v1",
        "sha256": hashlib.sha256(canonical_lines.encode("utf-8")).hexdigest(),
        "file_count": len(records),
        "files": records,
    }


def _test_snapshot(project: Path) -> dict[str, Any]:
    records = [
        _relative_record(
            project,
            _regular(project, project / relative, label=f"F0 test {relative}"),
        )
        for relative in F0_TEST_FILES
    ]
    return {
        "sha256": _canonical_sha256(records),
        "file_count": len(records),
        "files": records,
    }


def _load_json(path: Path, *, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise F0Error(f"could not read {label} as UTF-8 JSON: {path}: {error}") from error
    if not isinstance(value, Mapping):
        raise F0Error(f"{label} must be a JSON object")
    _reject_nonfinite(value, label=label)
    return value


def _environment_snapshot(
    project: Path,
    marker_path: Path,
    *,
    expected_bundle_sha256: str,
) -> dict[str, Any]:
    marker_path = _regular(project, marker_path, label="core environment marker")
    marker = _load_json(marker_path, label="core environment marker")
    if (
        marker.get("schema_version") != "PETCT-NNUNET-ENV-MARKER-v1.0"
        or marker.get("status") != "ENVIRONMENT_EVIDENCE_COMPLETE"
        or marker.get("bundle_sha256") != expected_bundle_sha256
    ):
        raise F0Error("core environment marker does not bind the expected immutable bundle")
    bundle_path_raw = marker.get("bundle_path")
    receipt_path_raw = marker.get("receipt_path")
    if not isinstance(bundle_path_raw, str) or not isinstance(receipt_path_raw, str):
        raise F0Error("core environment marker omits bundle/receipt paths")
    bundle_path = _directory(project, Path(bundle_path_raw), label="environment bundle")
    if bundle_path.name != expected_bundle_sha256:
        raise F0Error("environment bundle directory name differs from bundle SHA-256")
    bundle_manifest = _regular(
        project, bundle_path / "bundle.json", label="environment bundle manifest"
    )
    if _sha256(bundle_manifest) != marker.get("bundle_manifest_sha256"):
        raise F0Error("environment bundle manifest hash mismatch")
    bundle = _load_json(bundle_manifest, label="environment bundle manifest")
    unsigned_bundle = {key: value for key, value in bundle.items() if key != "bundle_sha256"}
    if (
        bundle.get("schema_version") != "PETCT-NNUNET-ENV-EVIDENCE-BUNDLE-v1.0"
        or bundle.get("status") != "ENVIRONMENT_EVIDENCE_COMPLETE"
        or bundle.get("bundle_sha256") != expected_bundle_sha256
        or _canonical_sha256(unsigned_bundle) != expected_bundle_sha256
    ):
        raise F0Error("environment bundle manifest content is invalid")
    file_records = bundle.get("files")
    if not isinstance(file_records, list) or not file_records:
        raise F0Error("environment bundle has no file inventory")
    for index, record in enumerate(file_records):
        if not isinstance(record, Mapping):
            raise F0Error(f"environment bundle file record {index} is invalid")
        name = record.get("name")
        if (
            not isinstance(name, str)
            or not name
            or Path(name).name != name
            or name in {".", ".."}
        ):
            raise F0Error(f"environment bundle file name is unsafe: {name!r}")
        artifact = _regular(
            project, bundle_path / name, label=f"environment bundle file {name}"
        )
        if (
            _sha256(artifact) != record.get("sha256")
            or artifact.stat().st_size != record.get("bytes")
        ):
            raise F0Error(f"environment bundle file record mismatch: {name}")
    receipt_path = _regular(
        project, Path(receipt_path_raw), label="core environment receipt"
    )
    if (
        not receipt_path.is_relative_to(bundle_path)
        or _sha256(receipt_path) != marker.get("receipt_sha256")
    ):
        raise F0Error("core environment receipt is outside or differs from the bundle")
    receipt = _load_json(receipt_path, label="core environment receipt")
    if receipt.get("schema_version") != "PETCT-NNUNET-ENV-v1.1":
        raise F0Error("core environment receipt is not PETCT-NNUNET-ENV-v1.1")
    return {
        "marker": _absolute_record(marker_path),
        "marker_schema_version": marker["schema_version"],
        "bundle_sha256": expected_bundle_sha256,
        "bundle_manifest": _absolute_record(bundle_manifest),
        "receipt": _absolute_record(receipt_path),
        "receipt_schema_version": receipt["schema_version"],
    }


def _canonical_inputs(
    project: Path,
    *,
    experiment_config: Path,
    official_simulator: Path,
    official_metrics: Path,
    official_runtime_manifest: Path,
) -> dict[str, Any]:
    simulator = _regular(
        project, official_simulator, label="official AutoPET V simulator"
    )
    metrics = _regular(project, official_metrics, label="official AutoPET V metrics")
    runtime_manifest = _regular(
        project,
        official_runtime_manifest,
        label="official AutoPET V minimal runtime manifest",
    )
    if _sha256(runtime_manifest) != EXPECTED_AUTOPETV_RUNTIME_MANIFEST_SHA256:
        raise F0Error("official AutoPET V minimal runtime manifest SHA-256 mismatch")
    try:
        manifest = json.loads(runtime_manifest.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise F0Error("official AutoPET V runtime manifest is not UTF-8 JSON") from error
    if not isinstance(manifest, Mapping):
        raise F0Error("official AutoPET V runtime manifest must be an object")
    expected_header = {
        "schema_version": AUTOPETV_RUNTIME_SCHEMA,
        "status": AUTOPETV_RUNTIME_STATUS,
        "upstream_repository": "https://github.com/lab-midas/autoPETV",
        "upstream_commit": AUTOPETV_RUNTIME_COMMIT,
        "license": "Apache-2.0",
    }
    for key, expected in expected_header.items():
        if manifest.get(key) != expected:
            raise F0Error(f"official AutoPET V runtime manifest {key} mismatch")
    entries = manifest.get("files")
    if not isinstance(entries, list) or not all(
        isinstance(item, Mapping) for item in entries
    ):
        raise F0Error("official AutoPET V runtime manifest files must be objects")
    by_path = {str(item.get("path")): item for item in entries}
    if len(by_path) != len(entries) or tuple(sorted(by_path)) != AUTOPETV_RUNTIME_FILES:
        raise F0Error("official AutoPET V runtime manifest allowlist mismatch")
    if by_path["interactive/simulate_scribbles.py"].get("required_callable") != (
        "simulate_scribble_from_label"
    ):
        raise F0Error("official AutoPET V simulator callable contract mismatch")
    if by_path["metrics.py"].get("required_callable") != "MetricEvaluator":
        raise F0Error("official AutoPET V metrics callable contract mismatch")
    if "required_callable" in by_path["LICENSE"]:
        raise F0Error("official AutoPET V LICENSE has an unexpected callable")

    runtime_root = _directory(
        project, simulator.parent.parent, label="official AutoPET V minimal runtime root"
    )
    expected_paths = {
        "interactive/simulate_scribbles.py": simulator,
        "metrics.py": metrics,
        "LICENSE": _regular(
            project, runtime_root / "LICENSE", label="official AutoPET V LICENSE"
        ),
    }
    if simulator != runtime_root / "interactive" / "simulate_scribbles.py":
        raise F0Error("official AutoPET V simulator is outside its minimal runtime layout")
    if metrics != runtime_root / "metrics.py":
        raise F0Error("official AutoPET V metrics is outside its minimal runtime layout")
    observed_files: set[str] = set()
    observed_dirs: set[str] = set()
    for candidate in sorted(runtime_root.rglob("*"), key=lambda item: item.as_posix()):
        if candidate.is_symlink():
            raise F0Error("official AutoPET V minimal runtime contains a symlink")
        relative = candidate.relative_to(runtime_root).as_posix()
        if candidate.is_dir():
            observed_dirs.add(relative)
        elif candidate.is_file():
            observed_files.add(relative)
        else:
            raise F0Error("official AutoPET V minimal runtime has a non-regular entry")
    if tuple(sorted(observed_files)) != AUTOPETV_RUNTIME_FILES:
        raise F0Error("official AutoPET V minimal runtime file inventory mismatch")
    if observed_dirs != {"interactive"}:
        raise F0Error("official AutoPET V minimal runtime directory inventory mismatch")
    file_records: list[dict[str, Any]] = []
    for relative in AUTOPETV_RUNTIME_FILES:
        record = _absolute_record(expected_paths[relative])
        if record["sha256"] != by_path[relative].get("sha256"):
            raise F0Error(
                f"official AutoPET V minimal runtime hash mismatch: {relative}"
            )
        file_records.append({"relative_path": relative, **record})
    runtime_binding = {
        "manifest": _absolute_record(runtime_manifest),
        "runtime_root": str(runtime_root),
        "upstream_commit": AUTOPETV_RUNTIME_COMMIT,
        "license": "Apache-2.0",
        "files": file_records,
    }
    runtime_binding["bundle_sha256"] = _canonical_sha256(runtime_binding)
    return {
        "experiment_config": _absolute_record(
            _regular(project, experiment_config, label="experiment config")
        ),
        "official_simulator": _absolute_record(simulator),
        "official_metrics": _absolute_record(metrics),
        "official_autopetv_runtime": runtime_binding,
    }


def _fixed_pytest_command(project: Path) -> list[str]:
    # Relative test paths plus an exact cwd make the recorded command portable
    # only within the one resolved project root bound by the receipt.
    return [
        str(Path(sys.executable).resolve()),
        "-m",
        "pytest",
        "-q",
        "--disable-warnings",
        *F0_TEST_FILES,
    ]


TestRunner = Callable[[Sequence[str], Path], subprocess.CompletedProcess[str]]


def _run_fixed_tests(
    command: Sequence[str], project: Path
) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["PYTHONUTF8"] = "1"
    environment["PYTHONIOENCODING"] = "utf-8"
    return subprocess.run(
        list(command),
        cwd=project,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )


def _ensure_new(project: Path, path: Path, *, label: str) -> Path:
    candidate = path if path.is_absolute() else project / path
    _reject_symlink_chain(project, candidate, label=label)
    if candidate.exists() or candidate.is_symlink():
        raise F0Error(f"{label} already exists; refusing overwrite: {candidate}")
    parent = candidate.parent
    if parent.exists():
        _directory(project, parent, label=f"{label} parent")
    else:
        # Only the evidence parent is created; no scientific run root is used.
        parent.mkdir(parents=True, exist_ok=False)
        _directory(project, parent, label=f"{label} parent")
    return candidate.absolute()


def _write_new_text(path: Path, text: str) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o640)
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())


def issue_receipt(
    *,
    project_root: Path,
    experiment_config: Path,
    environment_marker: Path,
    official_simulator: Path,
    official_metrics: Path,
    official_runtime_manifest: Path,
    output: Path,
    test_log: Path,
    expected_env_bundle_sha256: str = EXPECTED_ENV_BUNDLE_SHA256,
    test_runner: TestRunner = _run_fixed_tests,
) -> dict[str, Any]:
    project = _project_root(project_root)
    if len(expected_env_bundle_sha256) != 64:
        raise F0Error("expected environment bundle SHA-256 is malformed")
    output_candidate = output if output.is_absolute() else project / output
    test_log_candidate = test_log if test_log.is_absolute() else project / test_log
    if output_candidate.absolute() == test_log_candidate.absolute():
        raise F0Error("F0 receipt and pytest log must be different paths")
    output = _ensure_new(project, output, label="F0 receipt")
    test_log = _ensure_new(project, test_log, label="F0 pytest log")
    source_before = _source_snapshot(project)
    tests_before = _test_snapshot(project)
    canonical = _canonical_inputs(
        project,
        experiment_config=experiment_config,
        official_simulator=official_simulator,
        official_metrics=official_metrics,
        official_runtime_manifest=official_runtime_manifest,
    )
    environment = _environment_snapshot(
        project,
        environment_marker,
        expected_bundle_sha256=expected_env_bundle_sha256,
    )
    command = _fixed_pytest_command(project)
    completed = test_runner(command, project)
    captured = completed.stdout or ""
    if completed.returncode != 0:
        raise F0Error(
            "fixed F0 pytest suite failed with exit code "
            f"{completed.returncode}; no readiness receipt was published\n{captured}"
        )
    source_after = _source_snapshot(project)
    tests_after = _test_snapshot(project)
    canonical_after = _canonical_inputs(
        project,
        experiment_config=experiment_config,
        official_simulator=official_simulator,
        official_metrics=official_metrics,
        official_runtime_manifest=official_runtime_manifest,
    )
    environment_after = _environment_snapshot(
        project,
        environment_marker,
        expected_bundle_sha256=expected_env_bundle_sha256,
    )
    if (
        source_after != source_before
        or tests_after != tests_before
        or canonical_after != canonical
        or environment_after != environment
    ):
        raise F0Error(
            "source, tests, canonical inputs, or environment evidence changed "
            "while the fixed suite was running"
        )
    command_text = "argv=" + json.dumps(command, ensure_ascii=False) + "\n" + captured
    if not command_text.endswith("\n"):
        command_text += "\n"
    _write_new_text(test_log, command_text)
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "gate": "F0_ROUTE_A_TECHNICAL_READINESS",
        "closed_blocker_ids": list(BLOCKER_IDS),
        "project_root": str(project),
        "source_bundle": source_after,
        "test_evidence": {
            "fixed_suite": True,
            "command_argv": command,
            "cwd": str(project),
            "exit_code": 0,
            "encoding": "UTF-8",
            "selected_tests": tests_after,
            "log": _absolute_record(test_log),
        },
        "canonical_inputs": canonical,
        "environment": environment,
        "issued_at": datetime.now(timezone.utc).isoformat(),
        "scientific_execution": False,
        "scientific_result_claim": False,
        "claim_boundary": (
            "F0 closes code/data contract blockers only; it is not training, "
            "inference, evaluation, or evidence of method effectiveness."
        ),
    }
    _reject_nonfinite(payload, label="F0 receipt")
    _write_new_text(
        output,
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
    )
    return payload


def validate_receipt(
    *,
    receipt_path: Path,
    project_root: Path,
    experiment_config: Path,
    environment_marker: Path,
    official_simulator: Path,
    official_metrics: Path,
    official_runtime_manifest: Path,
    expected_env_bundle_sha256: str = EXPECTED_ENV_BUNDLE_SHA256,
) -> dict[str, Any]:
    project = _project_root(project_root)
    receipt_path = _regular(project, receipt_path, label="F0 receipt")
    payload = _load_json(receipt_path, label="F0 receipt")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise F0Error("F0 receipt schema mismatch")
    if payload.get("status") != "PASS" or payload.get("gate") != "F0_ROUTE_A_TECHNICAL_READINESS":
        raise F0Error("F0 receipt is not a technical-readiness PASS")
    if payload.get("closed_blocker_ids") != list(BLOCKER_IDS):
        raise F0Error("F0 receipt does not close the exact seven blocker IDs")
    if payload.get("project_root") != str(project):
        raise F0Error("F0 receipt belongs to a different project root")
    if payload.get("source_bundle") != _source_snapshot(project):
        raise F0Error("deployed Route A source bundle differs from F0")
    evidence = payload.get("test_evidence")
    if not isinstance(evidence, Mapping):
        raise F0Error("F0 receipt omits fixed pytest evidence")
    expected_command = _fixed_pytest_command(project)
    if (
        evidence.get("fixed_suite") is not True
        or evidence.get("command_argv") != expected_command
        or evidence.get("cwd") != str(project)
        or evidence.get("exit_code") != 0
        or evidence.get("encoding") != "UTF-8"
        or evidence.get("selected_tests") != _test_snapshot(project)
    ):
        raise F0Error("F0 fixed pytest evidence is incomplete or differs from deployment")
    log_record = evidence.get("log")
    if not isinstance(log_record, Mapping):
        raise F0Error("F0 receipt omits its pytest log record")
    log_path_raw = log_record.get("path")
    if not isinstance(log_path_raw, str):
        raise F0Error("F0 pytest log path is invalid")
    log_path = _regular(project, Path(log_path_raw), label="F0 pytest log")
    if _absolute_record(log_path) != dict(log_record):
        raise F0Error("F0 pytest log hash/size mismatch")
    if not log_path.read_text(encoding="utf-8").startswith(
        "argv=" + json.dumps(expected_command, ensure_ascii=False) + "\n"
    ):
        raise F0Error("F0 pytest log does not bind the fixed argv")
    expected_canonical = _canonical_inputs(
        project,
        experiment_config=experiment_config,
        official_simulator=official_simulator,
        official_metrics=official_metrics,
        official_runtime_manifest=official_runtime_manifest,
    )
    if payload.get("canonical_inputs") != expected_canonical:
        raise F0Error("canonical config or AutoPET V file binding differs from F0")
    expected_environment = _environment_snapshot(
        project,
        environment_marker,
        expected_bundle_sha256=expected_env_bundle_sha256,
    )
    if payload.get("environment") != expected_environment:
        raise F0Error("core environment binding differs from F0")
    if (
        payload.get("scientific_execution") is not False
        or payload.get("scientific_result_claim") is not False
    ):
        raise F0Error("F0 receipt crossed the scientific-result boundary")
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "gate": payload["gate"],
        "closed_blocker_ids": list(BLOCKER_IDS),
        "receipt": _absolute_record(receipt_path),
        "source_bundle_sha256": payload["source_bundle"]["sha256"],
        "environment_bundle_sha256": expected_env_bundle_sha256,
        "scientific_execution": False,
        "scientific_result_claim": False,
    }


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--experiment-config", type=Path, required=True)
    parser.add_argument("--environment-marker", type=Path, required=True)
    parser.add_argument("--official-simulator", type=Path, required=True)
    parser.add_argument("--official-metrics", type=Path, required=True)
    parser.add_argument("--official-runtime-manifest", type=Path, required=True)
    parser.add_argument(
        "--expected-env-bundle",
        default=EXPECTED_ENV_BUNDLE_SHA256,
        help="Exact immutable PETCT-NNUNET-ENV-v1.1 evidence bundle SHA-256.",
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    issue = subparsers.add_parser("issue", help="Run the fixed F0 tests and issue PASS.")
    _add_common_arguments(issue)
    issue.add_argument("--output", type=Path, required=True)
    issue.add_argument("--test-log", type=Path, required=True)
    validate = subparsers.add_parser("validate", help="Revalidate an existing F0 PASS.")
    _add_common_arguments(validate)
    validate.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "issue":
            issue_receipt(
                project_root=args.project_root,
                experiment_config=args.experiment_config,
                environment_marker=args.environment_marker,
                official_simulator=args.official_simulator,
                official_metrics=args.official_metrics,
                official_runtime_manifest=args.official_runtime_manifest,
                output=args.output,
                test_log=args.test_log,
                expected_env_bundle_sha256=args.expected_env_bundle,
            )
            summary = validate_receipt(
                receipt_path=args.output,
                project_root=args.project_root,
                experiment_config=args.experiment_config,
                environment_marker=args.environment_marker,
                official_simulator=args.official_simulator,
                official_metrics=args.official_metrics,
                official_runtime_manifest=args.official_runtime_manifest,
                expected_env_bundle_sha256=args.expected_env_bundle,
            )
        else:
            summary = validate_receipt(
                receipt_path=args.receipt,
                project_root=args.project_root,
                experiment_config=args.experiment_config,
                environment_marker=args.environment_marker,
                official_simulator=args.official_simulator,
                official_metrics=args.official_metrics,
                official_runtime_manifest=args.official_runtime_manifest,
                expected_env_bundle_sha256=args.expected_env_bundle,
            )
    except (F0Error, OSError, subprocess.SubprocessError) as error:
        parser.exit(1, f"F0 readiness validation failed: {error}\n")
    print(json.dumps(summary, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
