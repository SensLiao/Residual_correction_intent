#!/usr/bin/env python3
"""Delete exact superseded PET/CT artifacts only after R13 gates pass."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any, Mapping, Sequence

SCRIPTS = Path(__file__).resolve().parents[1]
if str(SCRIPTS) not in os.sys.path:
    os.sys.path.insert(0, str(SCRIPTS))

from common.petct_mainline_lineage import (  # noqa: E402
    LineageContractError,
    validate_r13_data_ready,
)


CONFIRMATION = "DELETE_SUPERSEDED_R13_LEGACY"
SMALL_EVIDENCE_LIMIT = 1024 * 1024
EVIDENCE_TOKENS = (
    "ready",
    "receipt",
    "summary",
    "status",
    "done",
    "fail",
    "launcher",
    "deployment",
    "audit",
)


class CleanupContractError(RuntimeError):
    pass


def _load_json(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise CleanupContractError(f"{label} is missing")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CleanupContractError(f"{label} is invalid JSON") from exc
    if not isinstance(value, dict):
        raise CleanupContractError(f"{label} must be a JSON object")
    return value


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _record(path: Path) -> dict[str, Any]:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha(path)}


def resolve_cleanup_targets(
    project_root: Path, plan: Mapping[str, Any]
) -> list[Path]:
    project_root = project_root.resolve()
    raw_targets = plan.get("delete_after_r13_gates")
    forbidden = plan.get("delete_gate", {}).get("forbidden_targets")
    if not isinstance(raw_targets, list) or not raw_targets:
        raise CleanupContractError("cleanup plan has no explicit targets")
    if not isinstance(forbidden, list) or not forbidden:
        raise CleanupContractError("cleanup plan has no forbidden-target firewall")
    forbidden_paths = []
    for raw in forbidden:
        if not isinstance(raw, str) or not raw or raw.startswith("/"):
            continue
        forbidden_paths.append((project_root / raw).resolve())
    resolved = []
    for item in raw_targets:
        raw = item.get("path") if isinstance(item, Mapping) else None
        if not isinstance(raw, str) or not raw:
            raise CleanupContractError("cleanup target path is missing")
        relative = Path(raw)
        if relative.is_absolute() or ".." in relative.parts:
            raise CleanupContractError("cleanup target must be project-relative")
        target = (project_root / relative).resolve()
        if not target.is_relative_to(project_root) or target == project_root:
            raise CleanupContractError("cleanup target escapes project root")
        if any(target == blocked or target.is_relative_to(blocked) for blocked in forbidden_paths):
            raise CleanupContractError(f"cleanup target is forbidden: {raw}")
        if target.is_symlink() or not target.is_dir():
            raise CleanupContractError(f"cleanup target is missing/non-directory: {raw}")
        resolved.append(target)
    if len(resolved) != len(set(resolved)):
        raise CleanupContractError("cleanup plan contains duplicate targets")
    return resolved


def validate_cleanup_gates(r13_root: Path, smoke_root: Path) -> dict[str, Any]:
    data_ready = r13_root / "R13-main" / "data-ready.json"
    if not data_ready.is_file():
        raise CleanupContractError("R13 data-ready gate is missing")
    try:
        validated = validate_r13_data_ready(data_ready)
    except LineageContractError as exc:
        raise CleanupContractError(f"R13 data-ready gate failed: {exc}") from exc
    for name in ("gate0a.done", "gate0b.done", "gate0c.done", "smoke.done"):
        path = smoke_root / "state" / name
        if not path.is_file() or path.is_symlink():
            raise CleanupContractError(f"R13 smoke gate is missing: {name}")
        document = _load_json(path, name)
        if document.get("status") != "PASS":
            raise CleanupContractError(f"R13 smoke gate is not PASS: {name}")
    return validated


def _active_reference_audit(project_root: Path, targets: Sequence[Path]) -> None:
    active_names = (
        "launch_petct_m0_v6_oof.sh",
        "launch_petct_m0_v6_oof_when_free.sh",
        "launch_petct_r13_mainline.sh",
        "launch_petct_r13_effect_smoke.sh",
        "launch_petct_r13_effect_val.sh",
        "watch_petct_r13_pipeline.sh",
    )
    tokens = {target.name for target in targets}
    root = project_root / "scripts" / "orchestration"
    for name in active_names:
        text = (root / name).read_text(encoding="utf-8")
        hits = sorted(token for token in tokens if token in text)
        if hits:
            raise CleanupContractError(
                f"active launcher {name} still references cleanup targets: {hits}"
            )


def _live_dependencies(targets: Sequence[Path]) -> list[dict[str, str]]:
    proc = Path("/proc")
    if not proc.is_dir():
        return []
    target_text = [str(path) for path in targets]
    found = []
    for process in proc.glob("[0-9]*"):
        if process.name == str(os.getpid()):
            continue
        try:
            parts = [
                value.decode(errors="replace")
                for value in (process / "cmdline").read_bytes().split(b"\0")
                if value
            ]
        except (OSError, PermissionError):
            parts = []
        command = " ".join(parts)
        if any(target in command for target in target_text):
            found.append({"pid": process.name, "kind": "cmdline", "value": command})
            continue
        fd_root = process / "fd"
        try:
            descriptors = list(fd_root.iterdir())
        except (OSError, PermissionError):
            continue
        for descriptor in descriptors:
            try:
                linked = str(descriptor.resolve())
            except (OSError, PermissionError):
                continue
            if any(linked == target or linked.startswith(target + os.sep) for target in target_text):
                found.append({"pid": process.name, "kind": "fd", "value": linked})
                break
    return found


def _tree_bytes(root: Path) -> int:
    total = 0
    for directory, _, files in os.walk(root):
        for name in files:
            try:
                total += (Path(directory) / name).stat().st_size
            except OSError:
                pass
    return total


def _capsule(targets: Sequence[Path], project_root: Path, capsule_root: Path) -> dict[str, Any]:
    if capsule_root.exists() or capsule_root.is_symlink():
        raise CleanupContractError("capsule root already exists")
    capsule_root.mkdir(parents=True, mode=0o700)
    target_records = []
    for target in targets:
        relative_target = target.relative_to(project_root)
        copied = []
        candidates = []
        for path in target.rglob("*"):
            if path.is_symlink() or not path.is_file():
                continue
            relative = path.relative_to(target)
            lowered = path.name.casefold()
            if path.stat().st_size <= SMALL_EVIDENCE_LIMIT and (
                len(relative.parts) <= 2
                or any(token in lowered for token in EVIDENCE_TOKENS)
            ):
                candidates.append(path)
        for path in sorted(candidates)[:200]:
            destination = capsule_root / relative_target / path.relative_to(target)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
            copied.append(_record(destination))
        log_tails = []
        for log in sorted(target.glob("logs/*.log"))[:20]:
            try:
                lines = log.read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError:
                continue
            tail = capsule_root / relative_target / "log-tails" / (log.name + ".tail.txt")
            tail.parent.mkdir(parents=True, exist_ok=True)
            tail.write_text("\n".join(lines[-200:]) + "\n", encoding="utf-8")
            log_tails.append(_record(tail))
        target_records.append(
            {
                "target": str(target),
                "bytes_before": _tree_bytes(target),
                "copied_evidence": copied,
                "log_tails": log_tails,
            }
        )
    manifest = {
        "schema_version": "PETCT-R13-LEGACY-CAPSULE-v1.0",
        "status": "PASS",
        "project_root": str(project_root),
        "targets": target_records,
    }
    manifest_path = capsule_root / "CAPSULE_MANIFEST.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.chmod(manifest_path, 0o600)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--r13-root", type=Path, required=True)
    parser.add_argument("--smoke-root", type=Path, required=True)
    parser.add_argument("--capsule-root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--execute-confirmation", required=True)
    args = parser.parse_args(argv)
    if args.execute_confirmation != CONFIRMATION:
        parser.error("exact cleanup confirmation is required")
    if args.receipt.exists() or args.receipt.is_symlink():
        parser.error("cleanup receipt already exists")
    project_root = args.project_root.resolve()
    plan = _load_json(args.plan, "server folder classification")
    validate_cleanup_gates(args.r13_root.resolve(), args.smoke_root.resolve())
    targets = resolve_cleanup_targets(project_root, plan)
    _active_reference_audit(project_root, targets)
    dependencies = _live_dependencies(targets)
    if dependencies:
        raise CleanupContractError(f"live cleanup dependencies exist: {dependencies}")
    capsule = _capsule(targets, project_root, args.capsule_root.resolve())
    journal = args.capsule_root.resolve() / "CLEANUP_JOURNAL.jsonl"
    reclaimed = 0
    with journal.open("x", encoding="utf-8") as stream:
        for target, record in zip(targets, capsule["targets"]):
            before = int(record["bytes_before"])
            stream.write(json.dumps({"target": str(target), "state": "DELETING", "bytes": before}) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
            shutil.rmtree(target)
            if target.exists() or target.is_symlink():
                raise CleanupContractError(f"cleanup target survived deletion: {target}")
            reclaimed += before
            stream.write(json.dumps({"target": str(target), "state": "DELETED", "bytes": before}) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
    receipt = {
        "schema_version": "PETCT-R13-LEGACY-CLEANUP-v1.0",
        "status": "PASS",
        "reclaimed_bytes": reclaimed,
        "capsule_manifest": _record(args.capsule_root.resolve() / "CAPSULE_MANIFEST.json"),
        "journal": _record(journal),
        "deleted_targets": [str(path) for path in targets],
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    with args.receipt.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({"status": "PASS", "reclaimed_bytes": reclaimed}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
