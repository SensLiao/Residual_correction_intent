#!/usr/bin/env python3
"""Fail-closed authorization for the W2.1 Binary/EDT official test run.

The gate binds the canonical clean learning split, the frozen 91-case
inventory, explicit model/checkpoint provenance, pinned official autoPET V
code, and one no-clobber output root.  It deliberately does not open any
CT/PET/GT NIfTI leaf while granting or consuming access.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


GRANT_SCHEMA = "PETCT-W21-OFFICIAL-TEST-GRANT-v2.0"
LEDGER_SCHEMA = "PETCT-W21-OFFICIAL-TEST-CONSUMPTION-v2.0"
RECEIPT_SCHEMA = "PETCT-W21-OFFICIAL-TEST-RECEIPT-v2.0"
CONFIRMATION = "I_AUTHORIZE_W21_BINARY_EDT_OFFICIAL_5_CORRECTION_TEST"
OFFICIAL_AUTOPETV_REPOSITORY = "https://github.com/lab-midas/autoPETV"
OFFICIAL_AUTOPETV_COMMIT = "4a2026866bfacc812492cfc7e6a8c54ac3c4f703"
OFFICIAL_METRICS_SHA256 = (
    "93e303219deb46b10fc5e5532873a42745aec1ecd6f78335f36cebba62104b83"
)
OFFICIAL_SIMULATOR_SHA256 = (
    "a2124e8aa4207e53ac93259214a35b7cf74626f83ab164e519769f86557d7cd2"
)
EXPECTED_LEARNING_SPLIT_SHA256 = (
    "2d428913bbbb142d45258072ca11a72fdeebacea3221389127b17b53fa446f0b"
)
MODEL_PROVENANCE_SCHEMA = "PETCT-W21-MODEL-PROVENANCE-v1.0"
EXPECTED_MODEL_FOLD = 0
EXPECTED_CHECKPOINT_NAME = "checkpoint_final.pth"
EXPECTED_CHECKPOINT_SELECTION = "fixed-final-no-post-test-selection"
PROTOCOL = {
    "correction_rounds": 5,
    "evaluation_states": 6,
    "scribbles_accumulate": True,
    "scribbles_per_round": 1,
    "strategy_assignment": "case-level-hash-rank-round-robin-equal-thirds-v1",
    "strategy_assignment_relation": (
        "project-frozen-balanced-realization-of-official-equal-thirds-design;"
        "not-upstream-verbatim"
    ),
    "strategy_salt": "PETCT-OFFICIAL-EQUAL-THIRDS-v1",
    "simulator_seed": 42,
    "metric_grid": "original-3d",
    "auc": "per-case-numpy-trapz-over-state-index-0-through-5",
    "dmm_connectivity": 18,
    "dmm_iou_threshold": 0.1,
    "rank_score": "0.5*mean_auc_dice+0.5*mean_auc_dmm",
    "empty_gt_policy": (
        "exclude-dice-dmm-auc-retain-official-component-fpv-fnv-and-raw-voxel-volumes"
    ),
}
STRATEGIES = ("centerline", "random", "boundary")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
EXPECTED_CASES = 597
EXPECTED_PATIENTS = 378
EXPECTED_TEST_CASES = 91
EXPECTED_TEST_PATIENTS = 57
EXPECTED_LEARNING_CASES = 506
EXPECTED_LEARNING_PATIENTS = 321
MODEL_FILES = (
    "dataset.json",
    "plans.json",
    "fold_0/checkpoint_final.pth",
    "training_provenance.json",
)


class W21AccessError(RuntimeError):
    """Raised when any frozen W2.1 test-access binding is invalid."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _regular(path: Path, *, label: str) -> Path:
    raw = Path(path)
    if raw.is_symlink() or not raw.is_file():
        raise W21AccessError(f"{label} must be a non-symlink regular file: {raw}")
    return raw.resolve()


def _record(path: Path, *, label: str) -> dict[str, Any]:
    resolved = _regular(path, label=label)
    return {
        "path": str(resolved),
        "size": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def _validate_record(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {"path", "size", "sha256"}:
        raise W21AccessError(f"{label} file record is invalid")
    if (
        not isinstance(value["path"], str)
        or isinstance(value["size"], bool)
        or not isinstance(value["size"], int)
        or value["size"] < 0
        or not isinstance(value["sha256"], str)
        or not HEX64.fullmatch(value["sha256"])
    ):
        raise W21AccessError(f"{label} file record fields are invalid")
    current = _record(Path(value["path"]), label=label)
    if current != dict(value):
        raise W21AccessError(f"{label} changed after authorization")
    return current


def _load_json(path: Path, *, label: str) -> tuple[Path, dict[str, Any]]:
    resolved = _regular(path, label=label)
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise W21AccessError(f"{label} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise W21AccessError(f"{label} must be a JSON object")
    return resolved, value


def _load_identity(path: Path) -> list[dict[str, Any]]:
    resolved = _regular(path, label="source identity manifest")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(resolved.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise W21AccessError(
                f"source identity line {line_number} is invalid JSON"
            ) from exc
        required = {
            "case_id",
            "patient_id",
            "held_out_fold",
            "ct_path",
            "pet_path",
            "gt_path",
            "truth_materialization",
        }
        if not isinstance(row, dict) or set(row) != required:
            raise W21AccessError("source identity row contract is invalid")
        if row["truth_materialization"] != "IDENTITY_ONLY":
            raise W21AccessError("source identity unexpectedly materializes truth")
        if not all(isinstance(row[key], str) and row[key] for key in required - {"held_out_fold"}):
            raise W21AccessError("source identity row has an empty string field")
        if row["patient_id"] != row["patient_id"].casefold():
            raise W21AccessError("source identity patient id is not canonical")
        if isinstance(row["held_out_fold"], bool) or row["held_out_fold"] not in range(5):
            raise W21AccessError("source identity held-out fold is invalid")
        rows.append(row)
    case_ids = [row["case_id"] for row in rows]
    if len(rows) != EXPECTED_CASES or case_ids != sorted(set(case_ids)):
        raise W21AccessError("source identity must contain 597 sorted unique cases")
    if len({row["patient_id"] for row in rows}) != EXPECTED_PATIENTS:
        raise W21AccessError("source identity must contain 378 patients")
    return rows


def _learning_split_index(learning_split: Path) -> dict[str, Any]:
    """Validate and index the frozen 597-case patient-level split."""

    _, split = _load_json(learning_split, label="learning split")
    if (
        split.get("schema_version") != "PETCT-LEARNING-SPLIT-v1.0"
        or split.get("status") != "FROZEN_BEFORE_MODEL_SELECTION"
        or split.get("case_count") != EXPECTED_CASES
        or split.get("patient_count") != EXPECTED_PATIENTS
    ):
        raise W21AccessError("learning split is not the frozen 597-case contract")
    patients = split.get("patients")
    if not isinstance(patients, list) or len(patients) != EXPECTED_PATIENTS:
        raise W21AccessError("learning split patient inventory is invalid")

    case_to_patient: dict[str, str] = {}
    case_to_partition: dict[str, str] = {}
    patient_to_partition: dict[str, str] = {}
    patient_to_cases: dict[str, list[str]] = {}
    partition_case_counts = {"train": 0, "val": 0, "test": 0}
    partition_patient_counts = {"train": 0, "val": 0, "test": 0}
    for patient in patients:
        if not isinstance(patient, Mapping):
            raise W21AccessError("learning split patient row is invalid")
        patient_id = patient.get("patient_id")
        partition = patient.get("partition")
        case_ids = patient.get("case_ids")
        if (
            not isinstance(patient_id, str)
            or not patient_id
            or patient_id != patient_id.casefold()
            or patient_id in patient_to_partition
            or partition not in partition_case_counts
            or not isinstance(case_ids, list)
            or not case_ids
        ):
            raise W21AccessError("learning split patient row fields are invalid")
        normalized_cases: list[str] = []
        for case_id in case_ids:
            if (
                not isinstance(case_id, str)
                or not case_id
                or case_id in case_to_partition
            ):
                raise W21AccessError("learning split case inventory is invalid")
            case_to_patient[case_id] = patient_id
            case_to_partition[case_id] = str(partition)
            normalized_cases.append(case_id)
            partition_case_counts[str(partition)] += 1
        patient_to_partition[patient_id] = str(partition)
        patient_to_cases[patient_id] = sorted(normalized_cases)
        partition_patient_counts[str(partition)] += 1

    if len(case_to_partition) != EXPECTED_CASES:
        raise W21AccessError("learning split must enumerate exactly 597 unique cases")
    if split.get("case_counts") != partition_case_counts:
        raise W21AccessError("learning split case_counts differs from patient rows")
    if partition_case_counts["test"] != EXPECTED_TEST_CASES:
        raise W21AccessError("learning split must contain exactly 91 test cases")
    if sum(partition_case_counts[key] for key in ("train", "val")) != EXPECTED_LEARNING_CASES:
        raise W21AccessError("learning split must contain exactly 506 train/val cases")
    if partition_patient_counts["test"] != EXPECTED_TEST_PATIENTS:
        raise W21AccessError("learning split must contain exactly 57 test patients")
    if sum(partition_patient_counts[key] for key in ("train", "val")) != EXPECTED_LEARNING_PATIENTS:
        raise W21AccessError("learning split must contain exactly 321 train/val patients")
    return {
        "document": split,
        "case_to_patient": case_to_patient,
        "case_to_partition": case_to_partition,
        "patient_to_partition": patient_to_partition,
        "patient_to_cases": patient_to_cases,
        "partition_case_counts": partition_case_counts,
        "partition_patient_counts": partition_patient_counts,
    }


def build_clean_learning_inventory(learning_split: Path) -> dict[str, Any]:
    """Return the canonical train/val-only inventory without opening image leaves."""

    index = _learning_split_index(learning_split)
    cases = [
        {
            "case_id": case_id,
            "patient_id": index["case_to_patient"][case_id],
            "partition": index["case_to_partition"][case_id],
        }
        for case_id in sorted(index["case_to_partition"])
        if index["case_to_partition"][case_id] in {"train", "val"}
    ]
    patients = [
        {
            "patient_id": patient_id,
            "partition": index["patient_to_partition"][patient_id],
            "case_ids": index["patient_to_cases"][patient_id],
        }
        for patient_id in sorted(index["patient_to_partition"])
        if index["patient_to_partition"][patient_id] in {"train", "val"}
    ]
    if len(cases) != EXPECTED_LEARNING_CASES or len(patients) != EXPECTED_LEARNING_PATIENTS:
        raise W21AccessError("clean train/val inventory count is invalid")
    return {
        "case_count": len(cases),
        "patient_count": len(patients),
        "partitions": ["train", "val"],
        "cases": cases,
        "patients": patients,
        "case_inventory_sha256": _canonical_sha256(cases),
        "patient_inventory_sha256": _canonical_sha256(patients),
    }


def assign_case_level_strategies(case_ids: Sequence[str]) -> dict[str, Any]:
    """Freeze a deterministic case-level allocation with exact thirds up to one case."""

    if isinstance(case_ids, (str, bytes)) or not case_ids:
        raise W21AccessError("strategy allocation needs a non-empty case sequence")
    normalized = [str(case_id) for case_id in case_ids]
    if any(not case_id for case_id in normalized) or len(set(normalized)) != len(normalized):
        raise W21AccessError("strategy allocation case ids must be non-empty and unique")
    ranked = sorted(
        normalized,
        key=lambda case_id: (
            hashlib.sha256(
                f"{PROTOCOL['strategy_salt']}|{case_id}".encode("utf-8")
            ).hexdigest(),
            case_id,
        ),
    )
    by_case = {
        case_id: STRATEGIES[index % len(STRATEGIES)]
        for index, case_id in enumerate(ranked)
    }
    counts = {
        strategy: sum(value == strategy for value in by_case.values())
        for strategy in STRATEGIES
    }
    if max(counts.values()) - min(counts.values()) > 1:
        raise W21AccessError("case-level strategy allocation is not balanced by thirds")
    return {
        "strategy_by_case": by_case,
        "strategy_case_counts": counts,
        "assignment_sha256": _canonical_sha256(by_case),
    }


def build_test_inventory(
    identity_manifest: Path, learning_split: Path
) -> dict[str, Any]:
    rows = _load_identity(identity_manifest)
    split_index = _learning_split_index(learning_split)
    test_by_case = {
        case_id: split_index["case_to_patient"][case_id]
        for case_id, partition in split_index["case_to_partition"].items()
        if partition == "test"
    }
    identity = {row["case_id"]: row for row in rows}
    if set(identity) != set(split_index["case_to_partition"]):
        raise W21AccessError("learning split and source identity case inventories differ")
    for case_id, row in identity.items():
        if row["patient_id"] != split_index["case_to_patient"][case_id]:
            raise W21AccessError("learning split and source identity patient mapping differ")
    if len(test_by_case) != EXPECTED_TEST_CASES:
        raise W21AccessError("learning split test inventory is not exactly 91 source cases")
    allocation = assign_case_level_strategies(sorted(test_by_case))
    cases = []
    for case_id in sorted(test_by_case):
        row = identity[case_id]
        patient_id = test_by_case[case_id]
        cases.append(
            {
                "case_id": case_id,
                "patient_id": patient_id,
                "held_out_fold": row["held_out_fold"],
                "strategy": allocation["strategy_by_case"][case_id],
            }
        )
    if len({row["patient_id"] for row in cases}) != EXPECTED_TEST_PATIENTS:
        raise W21AccessError("test inventory must contain 57 patients")
    return {
        "case_count": len(cases),
        "patient_count": len({row["patient_id"] for row in cases}),
        "cases": cases,
        "strategy_case_counts": allocation["strategy_case_counts"],
        "strategy_assignment_sha256": allocation["assignment_sha256"],
        "case_inventory_sha256": _canonical_sha256(cases),
    }


def _expected_model_provenance(
    *,
    arm: str,
    split_record: Mapping[str, Any],
    clean_inventory: Mapping[str, Any],
    checkpoint_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_version": MODEL_PROVENANCE_SCHEMA,
        "arm": arm,
        "learning_split_sha256": split_record["sha256"],
        "clean_train_val_case_inventory_sha256": clean_inventory[
            "case_inventory_sha256"
        ],
        "clean_train_val_patient_inventory_sha256": clean_inventory[
            "patient_inventory_sha256"
        ],
        "training_partitions": ["train", "val"],
        "training_case_count": EXPECTED_LEARNING_CASES,
        "training_patient_count": EXPECTED_LEARNING_PATIENTS,
        "test_case_count_consumed": 0,
        "fold": EXPECTED_MODEL_FOLD,
        "checkpoint_name": EXPECTED_CHECKPOINT_NAME,
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_selection": EXPECTED_CHECKPOINT_SELECTION,
    }


def _model_binding(
    model_dir: Path,
    *,
    arm: str,
    split_record: Mapping[str, Any],
    clean_inventory: Mapping[str, Any],
) -> dict[str, Any]:
    raw = Path(model_dir)
    if raw.is_symlink() or not raw.is_dir():
        raise W21AccessError(f"{arm} model directory must be a real directory")
    root = raw.resolve()
    files = {
        relative: _record(root / relative, label=f"{arm} model {relative}")
        for relative in MODEL_FILES
    }
    _, observed_provenance = _load_json(
        root / "training_provenance.json", label=f"{arm} training provenance"
    )
    expected_provenance = _expected_model_provenance(
        arm=arm,
        split_record=split_record,
        clean_inventory=clean_inventory,
        checkpoint_sha256=files["fold_0/checkpoint_final.pth"]["sha256"],
    )
    if observed_provenance != expected_provenance:
        raise W21AccessError(f"{arm} training provenance is not the clean fold/checkpoint contract")
    return {
        "arm": arm,
        "model_dir": str(root),
        "checkpoint_name": EXPECTED_CHECKPOINT_NAME,
        "fold": EXPECTED_MODEL_FOLD,
        "training_provenance": expected_provenance,
        "files": files,
    }


def _validate_model_binding(
    value: Any,
    *,
    arm: str,
    split_record: Mapping[str, Any],
    clean_inventory: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or value.get("arm") != arm:
        raise W21AccessError(f"{arm} model binding is invalid")
    if (
        value.get("checkpoint_name") != EXPECTED_CHECKPOINT_NAME
        or value.get("fold") != EXPECTED_MODEL_FOLD
    ):
        raise W21AccessError(f"{arm} model binding changed checkpoint/fold")
    root = Path(str(value.get("model_dir") or ""))
    if root.is_symlink() or not root.is_dir() or str(root.resolve()) != str(root):
        raise W21AccessError(f"{arm} model directory binding is invalid")
    files = value.get("files")
    if not isinstance(files, Mapping) or set(files) != set(MODEL_FILES):
        raise W21AccessError(f"{arm} model file inventory is invalid")
    for relative in MODEL_FILES:
        record = _validate_record(files[relative], label=f"{arm} model {relative}")
        if Path(record["path"]) != root / relative:
            raise W21AccessError(f"{arm} model file escapes its bound directory")
    checkpoint = files["fold_0/checkpoint_final.pth"]
    expected_provenance = _expected_model_provenance(
        arm=arm,
        split_record=split_record,
        clean_inventory=clean_inventory,
        checkpoint_sha256=checkpoint["sha256"],
    )
    _, current_provenance = _load_json(
        root / "training_provenance.json", label=f"{arm} training provenance"
    )
    if value.get("training_provenance") != expected_provenance:
        raise W21AccessError(f"{arm} embedded training provenance is invalid")
    if current_provenance != expected_provenance:
        raise W21AccessError(f"{arm} training provenance changed or is invalid")
    return dict(value)


def _official_code_binding(
    *, simulator_script: Path, metric_script: Path
) -> dict[str, Any]:
    simulator = _record(simulator_script, label="official simulator")
    metrics = _record(metric_script, label="official metrics")
    if simulator["sha256"] != OFFICIAL_SIMULATOR_SHA256:
        raise W21AccessError("official simulator hash is not the pinned autoPET V file")
    if metrics["sha256"] != OFFICIAL_METRICS_SHA256:
        raise W21AccessError("official metrics hash is not the pinned autoPET V file")
    return {
        "repository": OFFICIAL_AUTOPETV_REPOSITORY,
        "commit": OFFICIAL_AUTOPETV_COMMIT,
        "provenance_kind": "exact-file-hashes-at-pinned-upstream-commit",
        "simulator": simulator,
        "metrics": metrics,
    }


def _validate_official_code_binding(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {
        "repository",
        "commit",
        "provenance_kind",
        "simulator",
        "metrics",
    }:
        raise W21AccessError("official autoPET V code binding is invalid")
    if (
        value.get("repository") != OFFICIAL_AUTOPETV_REPOSITORY
        or value.get("commit") != OFFICIAL_AUTOPETV_COMMIT
        or value.get("provenance_kind")
        != "exact-file-hashes-at-pinned-upstream-commit"
    ):
        raise W21AccessError("official autoPET V repository/commit binding is invalid")
    simulator = _validate_record(value.get("simulator"), label="official simulator")
    metrics = _validate_record(value.get("metrics"), label="official metrics")
    if simulator["sha256"] != OFFICIAL_SIMULATOR_SHA256:
        raise W21AccessError("official simulator differs from the pinned hash")
    if metrics["sha256"] != OFFICIAL_METRICS_SHA256:
        raise W21AccessError("official metrics differs from the pinned hash")
    return dict(value)


def _seal(payload: Mapping[str, Any], field: str) -> dict[str, Any]:
    value = dict(payload)
    value[field] = _canonical_sha256(value)
    return value


def _verify_seal(value: Mapping[str, Any], field: str, *, label: str) -> None:
    observed = value.get(field)
    core = {key: item for key, item in value.items() if key != field}
    if not isinstance(observed, str) or observed != _canonical_sha256(core):
        raise W21AccessError(f"{label} seal is invalid")


def _write_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(
        payload, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2
    ) + "\n"
    descriptor = os.open(str(target), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        try:
            target.unlink()
        except FileNotFoundError:
            pass
        raise


def create_grant(
    *,
    identity_manifest: Path,
    learning_split: Path,
    binary_model_dir: Path,
    edt_model_dir: Path,
    runner_script: Path,
    simulator_script: Path,
    metric_script: Path,
    access_script: Path,
    run_root: Path,
    ledger_root: Path,
    grant_path: Path,
    authorized_by: str,
    confirmation: str,
) -> dict[str, Any]:
    if authorized_by not in {"director", "director-delegated-codex"}:
        raise W21AccessError("authorized_by is invalid")
    if confirmation != CONFIRMATION:
        raise W21AccessError("exact W2.1 final-test confirmation is required")
    raw_run = Path(run_root)
    if raw_run.is_symlink() or not raw_run.is_dir():
        raise W21AccessError("run root must already be a real directory")
    run = raw_run.resolve()
    raw_ledger = Path(ledger_root)
    if raw_ledger.is_symlink() or not raw_ledger.is_dir():
        raise W21AccessError("ledger root must already be a real directory")
    ledger = raw_ledger.resolve()
    identity_record = _record(identity_manifest, label="source identity manifest")
    split_record = _record(learning_split, label="learning split")
    if split_record["sha256"] != EXPECTED_LEARNING_SPLIT_SHA256:
        raise W21AccessError("learning split hash is not the canonical clean split")
    inventory = build_test_inventory(identity_manifest, learning_split)
    clean_inventory = build_clean_learning_inventory(learning_split)
    outputs = {
        "binary": str(run / "binary"),
        "edt": str(run / "edt"),
        "summary": str(run / "W21_OFFICIAL_TEST_SUMMARY.json"),
    }
    if any(Path(path).exists() or Path(path).is_symlink() for path in outputs.values()):
        raise W21AccessError("a granted scientific output already exists")
    binding = {
        "protocol": PROTOCOL,
        "source_identity": identity_record,
        "learning_split": split_record,
        "canonical_clean_learning_split_sha256": EXPECTED_LEARNING_SPLIT_SHA256,
        "clean_learning_inventory": clean_inventory,
        "test_inventory": inventory,
        "models": {
            "binary": _model_binding(
                binary_model_dir,
                arm="binary",
                split_record=split_record,
                clean_inventory=clean_inventory,
            ),
            "edt": _model_binding(
                edt_model_dir,
                arm="edt",
                split_record=split_record,
                clean_inventory=clean_inventory,
            ),
        },
        "code": {
            "runner": _record(runner_script, label="W2.1 test runner"),
            "access": _record(access_script, label="W2.1 access gate"),
            "official_autopetv": _official_code_binding(
                simulator_script=simulator_script,
                metric_script=metric_script,
            ),
        },
        "run_root": str(run),
        "ledger_root": str(ledger),
        "outputs": outputs,
    }
    payload = _seal(
        {
            "schema_version": GRANT_SCHEMA,
            "status": "AUTHORIZED_NOT_CONSUMED",
            "authorized_by": authorized_by,
            "confirmation": confirmation,
            "binding": binding,
            "binding_sha256": _canonical_sha256(binding),
        },
        "grant_sha256",
    )
    _write_exclusive(grant_path, payload)
    return payload


def _validate_binding(binding: Any, *, require_outputs_absent: bool) -> dict[str, Any]:
    if not isinstance(binding, Mapping) or binding.get("protocol") != PROTOCOL:
        raise W21AccessError("W2.1 scientific binding protocol is invalid")
    identity = _validate_record(binding.get("source_identity"), label="source identity manifest")
    split = _validate_record(binding.get("learning_split"), label="learning split")
    if (
        split["sha256"] != EXPECTED_LEARNING_SPLIT_SHA256
        or binding.get("canonical_clean_learning_split_sha256")
        != EXPECTED_LEARNING_SPLIT_SHA256
    ):
        raise W21AccessError("W2.1 binding does not use the canonical clean split")
    inventory = build_test_inventory(Path(identity["path"]), Path(split["path"]))
    if binding.get("test_inventory") != inventory:
        raise W21AccessError("W2.1 test inventory changed after authorization")
    clean_inventory = build_clean_learning_inventory(Path(split["path"]))
    if binding.get("clean_learning_inventory") != clean_inventory:
        raise W21AccessError("W2.1 clean train/val inventory changed after authorization")
    models = binding.get("models")
    if not isinstance(models, Mapping) or set(models) != {"binary", "edt"}:
        raise W21AccessError("W2.1 model binding is invalid")
    for arm in ("binary", "edt"):
        _validate_model_binding(
            models[arm],
            arm=arm,
            split_record=split,
            clean_inventory=clean_inventory,
        )
    code = binding.get("code")
    if not isinstance(code, Mapping) or set(code) != {
        "runner",
        "access",
        "official_autopetv",
    }:
        raise W21AccessError("W2.1 code binding is invalid")
    for label in ("runner", "access"):
        value = code[label]
        _validate_record(value, label=f"W2.1 {label}")
    _validate_official_code_binding(code["official_autopetv"])
    run = Path(str(binding.get("run_root") or ""))
    ledger = Path(str(binding.get("ledger_root") or ""))
    if run.is_symlink() or not run.is_dir() or str(run.resolve()) != str(run):
        raise W21AccessError("W2.1 run root binding is invalid")
    if ledger.is_symlink() or not ledger.is_dir() or str(ledger.resolve()) != str(ledger):
        raise W21AccessError("W2.1 ledger root binding is invalid")
    expected_outputs = {
        "binary": str(run / "binary"),
        "edt": str(run / "edt"),
        "summary": str(run / "W21_OFFICIAL_TEST_SUMMARY.json"),
    }
    if binding.get("outputs") != expected_outputs:
        raise W21AccessError("W2.1 output paths differ from the bound run root")
    if require_outputs_absent and any(
        Path(path).exists() or Path(path).is_symlink() for path in expected_outputs.values()
    ):
        raise W21AccessError("W2.1 scientific output exists before consumption")
    return dict(binding)


def _validate_grant(path: Path, *, require_outputs_absent: bool) -> tuple[Path, dict[str, Any]]:
    resolved, grant = _load_json(path, label="W2.1 test grant")
    _verify_seal(grant, "grant_sha256", label="W2.1 test grant")
    if (
        grant.get("schema_version") != GRANT_SCHEMA
        or grant.get("status") != "AUTHORIZED_NOT_CONSUMED"
        or grant.get("confirmation") != CONFIRMATION
        or grant.get("authorized_by") not in {"director", "director-delegated-codex"}
    ):
        raise W21AccessError("W2.1 test grant envelope is invalid")
    binding = _validate_binding(
        grant.get("binding"), require_outputs_absent=require_outputs_absent
    )
    if grant.get("binding_sha256") != _canonical_sha256(binding):
        raise W21AccessError("W2.1 test grant binding hash is invalid")
    return resolved, grant


def consume_grant(*, grant_path: Path, receipt_path: Path) -> dict[str, Any]:
    grant_file, grant = _validate_grant(grant_path, require_outputs_absent=True)
    binding = dict(grant["binding"])
    receipt = Path(receipt_path).resolve()
    run = Path(binding["run_root"])
    if receipt == run or not receipt.is_relative_to(run):
        raise W21AccessError("W2.1 receipt must be inside its bound run root")
    key = _canonical_sha256(
        {
            "schema_version": LEDGER_SCHEMA,
            "source_identity_sha256": binding["source_identity"]["sha256"],
            "learning_split_sha256": binding["learning_split"]["sha256"],
            "test_inventory_sha256": binding["test_inventory"]["case_inventory_sha256"],
            "models": {
                arm: binding["models"][arm]["files"]["fold_0/checkpoint_final.pth"]["sha256"]
                for arm in ("binary", "edt")
            },
            "protocol": PROTOCOL,
        }
    )
    core = {
        "consumption_key": key,
        "grant": _record(grant_file, label="W2.1 test grant"),
        "binding": binding,
        "binding_sha256": grant["binding_sha256"],
        "receipt_path": str(receipt),
    }
    ledger_path = Path(binding["ledger_root"]) / f"{key}.json"
    ledger = _seal(
        {
            "schema_version": LEDGER_SCHEMA,
            "status": "CONSUMED",
            "consumption": core,
            "consumption_sha256": _canonical_sha256(core),
        },
        "ledger_sha256",
    )
    try:
        _write_exclusive(ledger_path, ledger)
    except FileExistsError as exc:
        raise W21AccessError("this W2.1 test binding was already consumed") from exc
    payload = _seal(
        {
            "schema_version": RECEIPT_SCHEMA,
            "status": "CONSUMED",
            "consumption": core,
            "consumption_sha256": _canonical_sha256(core),
            "global_ledger": _record(ledger_path, label="W2.1 consumption ledger"),
        },
        "receipt_sha256",
    )
    try:
        _write_exclusive(receipt, payload)
    except Exception as exc:
        raise W21AccessError(
            "W2.1 access was consumed but the run receipt could not be published"
        ) from exc
    return payload


def validate_receipt(path: Path) -> dict[str, Any]:
    resolved, receipt = _load_json(path, label="W2.1 test receipt")
    _verify_seal(receipt, "receipt_sha256", label="W2.1 test receipt")
    if receipt.get("schema_version") != RECEIPT_SCHEMA or receipt.get("status") != "CONSUMED":
        raise W21AccessError("W2.1 test receipt envelope is invalid")
    core = receipt.get("consumption")
    if not isinstance(core, Mapping) or receipt.get("consumption_sha256") != _canonical_sha256(core):
        raise W21AccessError("W2.1 test receipt core is invalid")
    if core.get("receipt_path") != str(resolved):
        raise W21AccessError("W2.1 test receipt path differs from its claim")
    grant_record = _validate_record(core.get("grant"), label="W2.1 test grant")
    _, grant = _validate_grant(Path(grant_record["path"]), require_outputs_absent=False)
    binding = _validate_binding(core.get("binding"), require_outputs_absent=False)
    if binding != grant["binding"] or core.get("binding_sha256") != grant["binding_sha256"]:
        raise W21AccessError("W2.1 receipt binding differs from its grant")
    ledger_record = _validate_record(
        receipt.get("global_ledger"), label="W2.1 consumption ledger"
    )
    _, ledger = _load_json(Path(ledger_record["path"]), label="W2.1 consumption ledger")
    _verify_seal(ledger, "ledger_sha256", label="W2.1 consumption ledger")
    if (
        ledger.get("schema_version") != LEDGER_SCHEMA
        or ledger.get("status") != "CONSUMED"
        or ledger.get("consumption") != core
        or ledger.get("consumption_sha256") != receipt["consumption_sha256"]
    ):
        raise W21AccessError("W2.1 global ledger differs from its receipt")
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    grant = commands.add_parser("grant")
    grant.add_argument("--identity-manifest", type=Path, required=True)
    grant.add_argument("--learning-split", type=Path, required=True)
    grant.add_argument("--binary-model-dir", type=Path, required=True)
    grant.add_argument("--edt-model-dir", type=Path, required=True)
    grant.add_argument("--runner-script", type=Path, required=True)
    grant.add_argument("--simulator-script", type=Path, required=True)
    grant.add_argument("--metric-script", type=Path, required=True)
    grant.add_argument("--run-root", type=Path, required=True)
    grant.add_argument("--ledger-root", type=Path, required=True)
    grant.add_argument("--grant", type=Path, required=True)
    grant.add_argument(
        "--authorized-by",
        choices=("director", "director-delegated-codex"),
        required=True,
    )
    grant.add_argument("--confirm", required=True)
    consume = commands.add_parser("consume")
    consume.add_argument("--grant", type=Path, required=True)
    consume.add_argument("--receipt", type=Path, required=True)
    validate = commands.add_parser("validate")
    validate.add_argument("--receipt", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "grant":
            payload = create_grant(
                identity_manifest=args.identity_manifest,
                learning_split=args.learning_split,
                binary_model_dir=args.binary_model_dir,
                edt_model_dir=args.edt_model_dir,
                runner_script=args.runner_script,
                simulator_script=args.simulator_script,
                metric_script=args.metric_script,
                access_script=Path(__file__),
                run_root=args.run_root,
                ledger_root=args.ledger_root,
                grant_path=args.grant,
                authorized_by=args.authorized_by,
                confirmation=args.confirm,
            )
        elif args.command == "consume":
            payload = consume_grant(grant_path=args.grant, receipt_path=args.receipt)
        else:
            payload = validate_receipt(args.receipt)
    except (W21AccessError, FileExistsError) as exc:
        parser.error(str(exc))
    sys.stdout.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
