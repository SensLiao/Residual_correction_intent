#!/usr/bin/env python3
"""Fail-closed lineage contract for the PET/CT R13 mainline.

The contract deliberately names one source and one active dataset.  Historical
D1/D2/D3, R12, and Gate 0-C artifacts cannot satisfy it, even when their files
still exist for provenance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


MAINLINE_SOURCE = "M0_V6_FIVEFOLD_OOF"
MAINLINE_DATASET_ID = "R13-main-single-round"
LINEAGE_SCHEMA = "PETCT-R13-LINEAGE-v1.0"
DATA_READY_SCHEMA = "PETCT-R13-DATA-READY-v1.0"
M0_V6_OOF_SCHEMA = "PETCT-M0-V6-OOF-READY-v1.0"
EPISODE_SCHEMA = "single_round_one_scribble_one_strategy_v1"
ALLOWED_STRATEGIES = {"centerline", "random", "boundary"}
ALLOWED_PARTITIONS = {"train", "val"}


class LineageContractError(RuntimeError):
    """Raised when an active consumer is not bound to R13/M0-v6."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise LineageContractError(f"{label} must be a regular non-symlink file")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LineageContractError(f"{label} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise LineageContractError(f"{label} must contain one JSON object")
    return value


def _load_jsonl(path: Path, *, label: str) -> list[dict[str, Any]]:
    if path.is_symlink() or not path.is_file():
        raise LineageContractError(f"{label} must be a regular non-symlink file")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise LineageContractError(
                    f"{label} line {line_number} is not valid JSON"
                ) from exc
            if not isinstance(row, dict):
                raise LineageContractError(
                    f"{label} line {line_number} is not an object"
                )
            rows.append(row)
    if not rows:
        raise LineageContractError(f"{label} is empty")
    return rows


def file_record(path: Path) -> dict[str, Any]:
    path = path.resolve()
    if path.is_symlink() or not path.is_file():
        raise LineageContractError(f"required file is missing: {path}")
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _validate_record(record: Any, *, label: str) -> Path:
    if not isinstance(record, Mapping):
        raise LineageContractError(f"{label} record is missing")
    raw = record.get("path")
    size = record.get("bytes")
    digest = record.get("sha256")
    if (
        not isinstance(raw, str)
        or not raw
        or isinstance(size, bool)
        or not isinstance(size, int)
        or size < 0
        or not isinstance(digest, str)
        or not re.fullmatch(r"[0-9a-f]{64}", digest)
    ):
        raise LineageContractError(f"{label} record is malformed")
    path = Path(raw)
    if path.is_symlink() or not path.is_file():
        raise LineageContractError(f"{label} target is missing")
    path = path.resolve()
    if path.stat().st_size != size or _sha256_file(path) != digest:
        raise LineageContractError(f"{label} record no longer matches its file")
    return path


def validate_m0_v6_oof_ready(path: Path) -> dict[str, Any]:
    """Validate the clean 506-case OOF envelope without opening image inputs."""

    path = path.resolve()
    ready = _load_json(path, label="M0 v6 OOF receipt")
    required = {
        "schema_version": M0_V6_OOF_SCHEMA,
        "status": "PASS",
        "source_m0_lineage": MAINLINE_SOURCE,
        "mainline_eligible": True,
        "lifecycle": "active",
        "patient_excluded": True,
        "locked_test_present": False,
        "case_count": 506,
        "patient_count": 321,
        "fold_count": 5,
        "checkpoint_selector": "checkpoint_final.pth",
    }
    for key, expected in required.items():
        if ready.get(key) != expected:
            raise LineageContractError(f"M0 v6 OOF field {key} is invalid")
    run_dir = Path(str(ready.get("run_dir") or ""))
    if run_dir.is_symlink() or not run_dir.is_dir():
        raise LineageContractError("M0 v6 OOF run root is missing")
    run_dir = run_dir.resolve()
    bundle_path = _validate_record(ready.get("run_receipt"), label="M0 v6 OOF bundle")
    if bundle_path.parent != run_dir:
        raise LineageContractError("M0 v6 OOF bundle escapes its run root")
    bundle = _load_json(bundle_path, label="M0 v6 OOF bundle")
    if (
        bundle.get("schema_version") != "PETCT-M0-V6-OOF-BUNDLE-v1.0"
        or bundle.get("status") != "PASS"
        or bundle.get("source_m0_lineage") != MAINLINE_SOURCE
        or bundle.get("locked_test_present") is not False
    ):
        raise LineageContractError("M0 v6 OOF bundle contract is invalid")
    cases = bundle.get("cases")
    checkpoints = bundle.get("checkpoints")
    if not isinstance(cases, list) or len(cases) != 506:
        raise LineageContractError("M0 v6 OOF must contain exactly 506 learning cases")
    if not isinstance(checkpoints, list) or len(checkpoints) != 5:
        raise LineageContractError("M0 v6 OOF must bind exactly five checkpoints")
    for fold, record in enumerate(checkpoints):
        if not isinstance(record, Mapping) or record.get("fold") != fold:
            raise LineageContractError("M0 v6 checkpoint fold inventory is invalid")
        _validate_record(record.get("checkpoint"), label=f"fold {fold} checkpoint")
    splits_final = bundle.get("splits_final")
    _validate_record(splits_final, label="M0 v6 OOF splits_final")
    source_manifest = bundle.get("source_manifest")
    _validate_record(source_manifest, label="M0 v6 OOF source manifest")
    seen: set[str] = set()
    patients: defaultdict[str, set[int]] = defaultdict(set)
    normalized: dict[str, dict[str, Any]] = {}
    for case in cases:
        if not isinstance(case, dict):
            raise LineageContractError("M0 v6 OOF case row is not an object")
        case_id = str(case.get("case_id") or "")
        patient_id = str(case.get("patient_id") or "")
        fold = case.get("held_out_fold")
        if (
            not case_id
            or case_id in seen
            or not patient_id
            or isinstance(fold, bool)
            or not isinstance(fold, int)
            or fold not in range(5)
        ):
            raise LineageContractError("M0 v6 OOF case identity is invalid")
        seen.add(case_id)
        patients[patient_id].add(fold)
        for key in ("mask", "foreground_probability"):
            record = case.get(key)
            if not isinstance(record, Mapping):
                raise LineageContractError(f"M0 v6 OOF {key} record is missing")
            raw = record.get("path")
            if not isinstance(raw, str) or Path(raw).is_absolute() or ".." in Path(raw).parts:
                raise LineageContractError(f"M0 v6 OOF {key} path is unsafe")
            size = record.get("bytes")
            digest = record.get("sha256")
            if (
                isinstance(size, bool)
                or not isinstance(size, int)
                or size <= 0
                or not isinstance(digest, str)
                or not re.fullmatch(r"[0-9a-f]{64}", digest)
            ):
                raise LineageContractError(f"M0 v6 OOF {key} record is malformed")
        for modality in ("ct", "pet", "gt"):
            if not re.fullmatch(r"[0-9a-f]{64}", str(case.get(f"input_{modality}_sha256") or "")):
                raise LineageContractError(f"M0 v6 OOF input {modality} hash is missing")
            size = case.get(f"input_{modality}_bytes")
            if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
                raise LineageContractError(f"M0 v6 OOF input {modality} size is invalid")
        normalized[case_id] = case
    if len(patients) != 321 or any(len(folds) != 1 for folds in patients.values()):
        raise LineageContractError("M0 v6 OOF patient/fold exclusion is invalid")
    return {
        "status": "PASS",
        "schema_version": M0_V6_OOF_SCHEMA,
        "source_m0_lineage": MAINLINE_SOURCE,
        "ready_path": str(path),
        "ready_sha256": _sha256_file(path),
        "run_dir": str(run_dir),
        "patient_excluded": True,
        "locked_test_present": False,
        "cases": normalized,
        "checkpoints": checkpoints,
        "splits_final": splits_final,
        "source_manifest": source_manifest,
    }


def validate_r13_lineage_receipt(path: Path) -> dict[str, Any]:
    path = path.resolve()
    receipt = _load_json(path, label="R13 lineage receipt")
    required = {
        "schema_version": LINEAGE_SCHEMA,
        "status": "PASS",
        "dataset_id": MAINLINE_DATASET_ID,
        "source_m0_lineage": MAINLINE_SOURCE,
        "mainline_eligible": True,
        "lifecycle": "active",
        "episode_schema": EPISODE_SCHEMA,
        "round_count": 1,
        "scribbles_per_episode": 1,
        "strategy_is_label": False,
        "partitions": ["train", "val"],
        "locked_test_present": False,
    }
    for key, expected in required.items():
        if receipt.get(key) != expected:
            if key == "source_m0_lineage":
                raise LineageContractError(
                    f"source_m0_lineage must equal {MAINLINE_SOURCE}"
                )
            raise LineageContractError(f"R13 lineage field {key} is invalid")
    for key in ("oof_ready", "learning_split", "experiment_config"):
        _validate_record(receipt.get(key), label=key)
    return {**receipt, "receipt_path": str(path), "receipt_sha256": _sha256_file(path)}


def validate_r13_data_ready(
    path: Path,
    *,
    rich_tensor_manifest: Path | None = None,
    learning_split: Path | None = None,
    experiment_config: Path | None = None,
) -> dict[str, Any]:
    path = path.resolve()
    ready = _load_json(path, label="R13 data-ready receipt")
    required = {
        "schema_version": DATA_READY_SCHEMA,
        "status": "PASS",
        "dataset_id": MAINLINE_DATASET_ID,
        "source_m0_lineage": MAINLINE_SOURCE,
        "mainline_eligible": True,
        "lifecycle": "active",
        "locked_test_present": False,
    }
    for key, expected in required.items():
        if ready.get(key) != expected:
            raise LineageContractError(f"R13 data-ready field {key} is invalid")
    lineage_path = _validate_record(
        ready.get("lineage_receipt"), label="R13 lineage receipt"
    )
    lineage = validate_r13_lineage_receipt(lineage_path)
    for label, supplied in (
        ("learning_split", learning_split),
        ("experiment_config", experiment_config),
    ):
        if supplied is not None:
            bound = Path(str(lineage[label]["path"])).resolve()
            if bound != supplied.resolve():
                raise LineageContractError(
                    f"R13 {label} differs from lineage receipt"
                )
    _validate_record(
        ready.get("program_manifest_receipt"), label="R13 program manifest receipt"
    )
    outputs = ready.get("outputs")
    if not isinstance(outputs, Mapping):
        raise LineageContractError("R13 data-ready outputs are missing")
    for key in (
        "inference_visible",
        "label_only",
        "audit_only",
        "rich_tensors",
        "candidate_summary",
        "pointer_summary",
    ):
        target = _validate_record(outputs.get(key), label=f"R13 {key}")
        if key == "rich_tensors" and rich_tensor_manifest is not None:
            if target != rich_tensor_manifest.resolve():
                raise LineageContractError(
                    "R13 rich tensor manifest differs from data-ready receipt"
                )
    return {
        **ready,
        "receipt_path": str(path),
        "receipt_sha256": _sha256_file(path),
        "validated_lineage": lineage,
    }


def validate_r13_program_rows(rows: Iterable[Mapping[str, Any]]) -> None:
    seen: set[str] = set()
    families: defaultdict[str, list[str]] = defaultdict(list)
    count = 0
    for row in rows:
        count += 1
        episode_id = str(row.get("episode_id") or "")
        family_id = str(row.get("episode_family_id") or "")
        strategy = str(row.get("strategy") or "")
        if not episode_id or episode_id in seen:
            raise LineageContractError("R13 episode IDs must be non-empty and unique")
        seen.add(episode_id)
        if row.get("source_m0_lineage") != MAINLINE_SOURCE:
            raise LineageContractError(f"every R13 row must bind {MAINLINE_SOURCE}")
        if row.get("partition") not in ALLOWED_PARTITIONS:
            raise LineageContractError("R13-main contains only TRAIN/VAL rows")
        if row.get("round_index") != 0:
            raise LineageContractError("R13-main is single-round; trajectory rows are separate")
        if row.get("scribble_count") != 1:
            raise LineageContractError("R13-main requires exactly one scribble per episode")
        if not family_id:
            raise LineageContractError("R13 episode_family_id is missing")
        if strategy not in ALLOWED_STRATEGIES:
            raise LineageContractError("R13 strategy is not a frozen simulator strategy")
        families[family_id].append(strategy)
    if count == 0:
        raise LineageContractError("R13 program manifest is empty")
    for family_id, strategies in families.items():
        if len(strategies) > 3:
            raise LineageContractError(
                f"R13 family {family_id} has at most three strategy siblings"
            )
        if len(strategies) != len(set(strategies)):
            raise LineageContractError(
                f"R13 family {family_id} repeats a strategy sibling"
            )


def validate_r13_training_binding(
    lineage_receipt: Path,
    manifest_receipt: Path,
    episodes: Path,
    labels: Path,
) -> dict[str, Any]:
    lineage = validate_r13_lineage_receipt(lineage_receipt)
    manifest = _load_json(manifest_receipt.resolve(), label="program manifest receipt")
    if (
        manifest.get("status") != "PASS"
        or manifest.get("source_m0_lineage") != MAINLINE_SOURCE
        or manifest.get("dataset_id") != MAINLINE_DATASET_ID
        or manifest.get("mainline_eligible") is not True
        or manifest.get("lifecycle") != "active"
        or manifest.get("locked_test_present") is not False
    ):
        raise LineageContractError("program manifest receipt is not R13 mainline eligible")
    bound_lineage = manifest.get("lineage_receipt")
    if not isinstance(bound_lineage, Mapping):
        raise LineageContractError("program manifest receipt lacks lineage binding")
    lineage_path = _validate_record(bound_lineage, label="bound lineage receipt")
    if (
        lineage_path != Path(lineage["receipt_path"])
        or bound_lineage.get("sha256") != lineage["receipt_sha256"]
    ):
        raise LineageContractError("program manifest binds a different lineage receipt")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, Mapping):
        raise LineageContractError("program manifest outputs are missing")
    for name, supplied in (("inference", episodes), ("labels", labels)):
        bound = _validate_record(outputs.get(name), label=f"program {name}")
        if bound != supplied.resolve():
            raise LineageContractError(f"program {name} path differs from receipt")
    validate_r13_program_rows(_load_jsonl(labels, label="R13 label manifest"))
    return lineage


def issue_r13_lineage_receipt(
    *,
    oof_ready: Path,
    learning_split: Path,
    experiment_config: Path,
    output: Path,
) -> dict[str, Any]:
    if output.exists() or output.is_symlink():
        raise LineageContractError("refusing to overwrite R13 lineage receipt")
    validate_m0_v6_oof_ready(oof_ready)
    payload = {
        "schema_version": LINEAGE_SCHEMA,
        "status": "PASS",
        "dataset_id": MAINLINE_DATASET_ID,
        "source_m0_lineage": MAINLINE_SOURCE,
        "mainline_eligible": True,
        "lifecycle": "active",
        "episode_schema": EPISODE_SCHEMA,
        "round_count": 1,
        "scribbles_per_episode": 1,
        "strategy_is_label": False,
        "partitions": ["train", "val"],
        "locked_test_present": False,
        "oof_ready": file_record(oof_ready),
        "learning_split": file_record(learning_split),
        "experiment_config": file_record(experiment_config),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    validate_r13_lineage_receipt(output)
    return payload


def seal_r13_data_ready(
    *,
    lineage_receipt: Path,
    manifest_receipt: Path,
    inference_manifest: Path,
    label_manifest: Path,
    audit_manifest: Path,
    rich_tensor_manifest: Path,
    candidate_summary: Path,
    pointer_summary: Path,
    output: Path,
) -> dict[str, Any]:
    lineage = validate_r13_training_binding(
        lineage_receipt, manifest_receipt, inference_manifest, label_manifest
    )
    if output.exists() or output.is_symlink():
        raise LineageContractError("refusing to overwrite R13 data-ready receipt")
    payload = {
        "schema_version": DATA_READY_SCHEMA,
        "status": "PASS",
        "dataset_id": MAINLINE_DATASET_ID,
        "source_m0_lineage": MAINLINE_SOURCE,
        "mainline_eligible": True,
        "lifecycle": "active",
        "locked_test_present": False,
        "lineage_receipt": file_record(lineage_receipt),
        "program_manifest_receipt": file_record(manifest_receipt),
        "outputs": {
            "inference_visible": file_record(inference_manifest),
            "label_only": file_record(label_manifest),
            "audit_only": file_record(audit_manifest),
            "rich_tensors": file_record(rich_tensor_manifest),
            "candidate_summary": file_record(candidate_summary),
            "pointer_summary": file_record(pointer_summary),
        },
        "row_count": len(_load_jsonl(label_manifest, label="R13 label manifest")),
        "lineage_receipt_sha256": lineage["receipt_sha256"],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    issue = commands.add_parser("issue")
    issue.add_argument("--oof-ready", type=Path, required=True)
    issue.add_argument("--learning-split", type=Path, required=True)
    issue.add_argument("--experiment-config", type=Path, required=True)
    issue.add_argument("--output", type=Path, required=True)
    validate = commands.add_parser("validate")
    validate.add_argument("--receipt", type=Path, required=True)
    validate_data = commands.add_parser("validate-data")
    validate_data.add_argument("--receipt", type=Path, required=True)
    seal = commands.add_parser("seal")
    seal.add_argument("--lineage-receipt", type=Path, required=True)
    seal.add_argument("--manifest-receipt", type=Path, required=True)
    seal.add_argument("--inference-manifest", type=Path, required=True)
    seal.add_argument("--label-manifest", type=Path, required=True)
    seal.add_argument("--audit-manifest", type=Path, required=True)
    seal.add_argument("--rich-tensor-manifest", type=Path, required=True)
    seal.add_argument("--candidate-summary", type=Path, required=True)
    seal.add_argument("--pointer-summary", type=Path, required=True)
    seal.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "issue":
        payload = issue_r13_lineage_receipt(
            oof_ready=args.oof_ready,
            learning_split=args.learning_split,
            experiment_config=args.experiment_config,
            output=args.output,
        )
    elif args.command == "validate":
        payload = validate_r13_lineage_receipt(args.receipt)
    elif args.command == "validate-data":
        payload = validate_r13_data_ready(args.receipt)
    else:
        payload = seal_r13_data_ready(
            lineage_receipt=args.lineage_receipt,
            manifest_receipt=args.manifest_receipt,
            inference_manifest=args.inference_manifest,
            label_manifest=args.label_manifest,
            audit_manifest=args.audit_manifest,
            rich_tensor_manifest=args.rich_tensor_manifest,
            candidate_summary=args.candidate_summary,
            pointer_summary=args.pointer_summary,
            output=args.output,
        )
    print(json.dumps({"status": "PASS", "dataset_id": payload["dataset_id"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
