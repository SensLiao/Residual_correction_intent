#!/usr/bin/env python3
"""Authorize exactly one test-only M0 OOF baseline metrics run.

This gate is intentionally narrower than the final P2T/editor test gate.  It
permits only materializing the frozen test truth and scoring the already
committed patient-excluded ``OOF_READY`` predictions with the pinned official
metric implementation.  It does not authorize inference, ensembling, residual
construction, model selection, P2T, or editor evaluation.

The grant binds every scientific input, both executable entry points, the
exact test case/patient inventory, the run root, and all final output paths.
Consumption is claimed in an independent project-global O_EXCL ledger whose
key excludes the grant and run paths, so copying a grant or choosing another
run directory cannot reset the exactly-once decision.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))
for support_dir in (SCRIPTS_ROOT / "baseline", SCRIPTS_ROOT / "data"):
    if str(support_dir) not in sys.path:
        sys.path.insert(0, str(support_dir))

from baseline.validate_petct_m0_oof import (  # noqa: E402
    ContractError,
    validate_oof_ready_receipt_only,
)
from common.petct_test_access import (  # noqa: E402
    ALLOWED_AUTHORIZERS,
    TestAccessError,
    _canonical_sha256,
    _file_record,
    _load_json,
    _sealed,
    _validate_config_and_split,
    _verify_seal,
    _write_json_exclusive,
)
from data.validate_petct_learning_split import (  # noqa: E402
    load_and_validate_learning_split,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
GLOBAL_LEDGER_ROOT = (
    PROJECT_ROOT
    / "records"
    / "test-access"
    / "m0-baseline-global-consumption-ledger"
)
LEDGER_RELATIVE_PARTS = (
    "records",
    "test-access",
    "m0-baseline-global-consumption-ledger",
)

GRANT_SCHEMA = "PETCT-M0-BASELINE-TEST-GRANT-v1.0"
LEDGER_SCHEMA = "PETCT-M0-BASELINE-TEST-GLOBAL-CONSUMPTION-v1.0"
RECEIPT_SCHEMA = "PETCT-M0-BASELINE-TEST-CONSUMED-v1.0"
EXPERIMENT_SCHEMA = "PETCT-ROUTE-A-EXPERIMENT-v2.0"
LEARNING_SPLIT_SCHEMA = "PETCT-LEARNING-SPLIT-v1.0"
OOF_READY_SCHEMA = "PETCT-M0-OOF-READY-v1.0"

SCOPE = "m0_oof_baseline_metrics"
PARTITION = "test"
AUTHORIZATION_KIND = "explicit-director-command-m0-baseline-test-metrics"
M0_TEST_CONFIRMATION = "I_AUTHORIZE_ONE_M0_OOF_BASELINE_TEST_METRICS_RUN"

EXPECTED_DATASET = "PSMA-PET-CT-Lesions-v3"
EXPECTED_CASES = 597
EXPECTED_PATIENTS = 378
EXPECTED_FOLDS = 5
EXPECTED_TEST_CASES = 91
EXPECTED_TEST_PATIENTS = 57
EXPECTED_TEST_FOLD_CASE_COUNTS = {"0": 20, "1": 14, "2": 23, "3": 19, "4": 15}
EXPECTED_PATIENT_COUNTS = {"train": 264, "val": 57, "test": 57}
EXPECTED_M0_ROLE = (
    "one OOF family: every case receives exactly one held-out-fold prediction"
)
EXPECTED_EVALUATOR_NAME = "evaluate_petct_m0_oof.py"
EXPECTED_RUNNER_NAME = "run_petct_m0_test_baseline.py"
PREDICTION_CONTRACT = (
    "one existing patient-excluded held-out-fold OOF prediction per case"
)
FORBIDDEN_DOWNSTREAM_SCOPES = [
    "residual_atlas",
    "scribble_generation",
    "p2t",
    "editor",
    "model_selection",
    "threshold_tuning",
]

IDENTITY_FIELDS = {
    "case_id",
    "patient_id",
    "held_out_fold",
    "ct_path",
    "pet_path",
    "gt_path",
    "truth_materialization",
}
IDENTITY_STATE = "IDENTITY_ONLY"
HEX_64 = re.compile(r"[0-9a-f]{64}")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _required_record_path(record: Any, *, label: str) -> Path:
    if not isinstance(record, Mapping):
        raise TestAccessError(f"{label} record is missing")
    raw = record.get("path")
    if not isinstance(raw, str) or not raw:
        raise TestAccessError(f"{label} record path is missing")
    return Path(raw)


def _load_identity_manifest(
    path: Path,
) -> tuple[Path, list[dict[str, Any]], dict[str, Any]]:
    path = Path(path)
    if path.is_symlink():
        raise TestAccessError("identity manifest must be a regular non-symlink file")
    path = path.resolve()
    if not path.is_file():
        raise TestAccessError("identity manifest must be a regular non-symlink file")
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise TestAccessError("identity manifest is not valid UTF-8 JSONL") from exc
    if not lines or any(not line.strip() for line in lines):
        raise TestAccessError("identity manifest must be non-empty JSONL without blanks")
    for line_number, line in enumerate(lines, start=1):
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise TestAccessError(
                f"identity manifest line {line_number} is invalid JSON"
            ) from exc
        if not isinstance(raw, dict) or set(raw) != IDENTITY_FIELDS:
            raise TestAccessError(
                f"identity manifest line {line_number} has an invalid field set"
            )
        case_id = raw.get("case_id")
        patient_id = raw.get("patient_id")
        fold = raw.get("held_out_fold")
        if (
            not isinstance(case_id, str)
            or not case_id
            or not isinstance(patient_id, str)
            or not patient_id
            or patient_id != patient_id.casefold()
            or isinstance(fold, bool)
            or not isinstance(fold, int)
            or fold not in range(EXPECTED_FOLDS)
            or raw.get("truth_materialization") != IDENTITY_STATE
        ):
            raise TestAccessError(
                f"identity manifest line {line_number} has invalid identity/fold/state"
            )
        for field in ("ct_path", "pet_path", "gt_path"):
            value = raw.get(field)
            if not isinstance(value, str) or not value or not Path(value).is_absolute():
                raise TestAccessError(
                    f"identity manifest line {line_number} requires absolute {field}"
                )
        rows.append(dict(raw))

    cases = [str(row["case_id"]) for row in rows]
    if len(rows) != EXPECTED_CASES or len(set(cases)) != EXPECTED_CASES:
        raise TestAccessError("identity manifest must contain exactly 597 unique cases")
    if cases != sorted(cases):
        raise TestAccessError("identity manifest must be sorted by case_id")
    patient_folds: dict[str, set[int]] = defaultdict(set)
    for row in rows:
        patient_folds[str(row["patient_id"])].add(int(row["held_out_fold"]))
    if len(patient_folds) != EXPECTED_PATIENTS:
        raise TestAccessError("identity manifest must contain exactly 378 patients")
    if any(len(folds) != 1 for folds in patient_folds.values()):
        raise TestAccessError("identity manifest assigns one patient to multiple folds")
    represented_folds = {int(row["held_out_fold"]) for row in rows}
    if represented_folds != set(range(EXPECTED_FOLDS)):
        raise TestAccessError("identity manifest does not represent all five folds")
    return path, rows, _file_record(path, label="identity manifest")


def _validate_current_experiment(config: Mapping[str, Any]) -> None:
    if config.get("schema_version") != EXPERIMENT_SCHEMA:
        raise TestAccessError("M0 baseline test gate requires the v2 experiment schema")
    dataset = config.get("dataset")
    m0 = config.get("m0")
    learning = dataset.get("learning_split") if isinstance(dataset, Mapping) else None
    expected_dataset = {
        "name": EXPECTED_DATASET,
        "cases": EXPECTED_CASES,
        "patients": EXPECTED_PATIENTS,
        "split_unit": "patient",
        "folds": EXPECTED_FOLDS,
    }
    if not isinstance(dataset, Mapping) or any(
        dataset.get(key) != value for key, value in expected_dataset.items()
    ):
        raise TestAccessError("M0 baseline test gate dataset contract is invalid")
    if (
        not isinstance(learning, Mapping)
        or learning.get("schema_version") != LEARNING_SPLIT_SCHEMA
        or learning.get("target_patient_counts") != EXPECTED_PATIENT_COUNTS
    ):
        raise TestAccessError("M0 baseline test gate learning-split contract is invalid")
    if not isinstance(m0, Mapping) or m0.get("role") != EXPECTED_M0_ROLE:
        raise TestAccessError("M0 role is not the patient-excluded OOF contract")


def _test_inventory(
    identity_rows: Sequence[Mapping[str, Any]],
    case_to_partition: Mapping[str, str],
) -> dict[str, Any]:
    selected = [
        {
            "case_id": str(row["case_id"]),
            "patient_id": str(row["patient_id"]),
            "held_out_fold": int(row["held_out_fold"]),
        }
        for row in identity_rows
        if case_to_partition.get(str(row["case_id"])) == PARTITION
    ]
    selected.sort(key=lambda row: row["case_id"])
    grouped: dict[str, dict[str, Any]] = {}
    for row in selected:
        patient_id = row["patient_id"]
        patient = grouped.setdefault(
            patient_id,
            {
                "patient_id": patient_id,
                "held_out_fold": row["held_out_fold"],
                "case_ids": [],
            },
        )
        if patient["held_out_fold"] != row["held_out_fold"]:
            raise TestAccessError("test patient is assigned to multiple held-out folds")
        patient["case_ids"].append(row["case_id"])
    patients = [grouped[patient_id] for patient_id in sorted(grouped)]
    if len(selected) != EXPECTED_TEST_CASES:
        raise TestAccessError("frozen test inventory must contain exactly 91 cases")
    if len(patients) != EXPECTED_TEST_PATIENTS:
        raise TestAccessError("frozen test inventory must contain exactly 57 patients")
    fold_case_counts = {
        str(fold): sum(1 for row in selected if row["held_out_fold"] == fold)
        for fold in range(EXPECTED_FOLDS)
    }
    if fold_case_counts != EXPECTED_TEST_FOLD_CASE_COUNTS:
        raise TestAccessError(
            "frozen test inventory held-out-fold counts must be 20/14/23/19/15"
        )
    return {
        "case_count": len(selected),
        "patient_count": len(patients),
        "held_out_fold_case_counts": fold_case_counts,
        "cases": selected,
        "patients": patients,
        "case_inventory_sha256": _canonical_sha256(selected),
        "patient_inventory_sha256": _canonical_sha256(patients),
    }


def _validate_oof_identity(
    oof_ready: Path,
    identity_rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    oof_record = _file_record(oof_ready, label="OOF_READY")
    try:
        validated = validate_oof_ready_receipt_only(Path(oof_ready))
    except (ContractError, RuntimeError, OSError, ValueError) as exc:
        raise TestAccessError(f"OOF_READY is invalid: {exc}") from exc
    expected_header = {
        "status": "PASS",
        "schema_version": OOF_READY_SCHEMA,
        "phase": "PATIENT_EXCLUDED_5FOLD_OOF",
        "patient_excluded": True,
    }
    if not isinstance(validated, Mapping) or any(
        validated.get(key) != value for key, value in expected_header.items()
    ):
        raise TestAccessError("OOF_READY is not a patient-excluded five-fold PASS")
    if validated.get("ready_path") != oof_record["path"]:
        raise TestAccessError("OOF_READY validator returned a different receipt path")
    if validated.get("ready_sha256") != oof_record["sha256"]:
        raise TestAccessError("OOF_READY validator returned a different receipt hash")
    cases = validated.get("cases")
    identity = {str(row["case_id"]): row for row in identity_rows}
    if not isinstance(cases, Mapping) or set(cases) != set(identity):
        raise TestAccessError("OOF_READY inventory differs from the identity manifest")
    for case_id, source in identity.items():
        oof = cases.get(case_id)
        if not isinstance(oof, Mapping):
            raise TestAccessError("OOF_READY contains a non-object case record")
        if (
            oof.get("patient_id") != source["patient_id"]
            or oof.get("held_out_fold") != source["held_out_fold"]
        ):
            raise TestAccessError("OOF_READY case/patient/fold identity differs")
    return dict(validated), oof_record


def _canonical_ledger_root_from_oof_record(oof_record: Mapping[str, Any]) -> Path:
    """Derive the one project-global ledger root from canonical OOF_READY."""

    oof_path = _required_record_path(oof_record, label="OOF_READY").resolve()
    if (
        oof_path.name != "OOF_READY.json"
        or oof_path.parent.name != "manifests"
        or oof_path.parent.parent.name != "nnunet"
    ):
        raise TestAccessError(
            "OOF_READY must live at <project>/nnunet/manifests/OOF_READY.json"
        )
    project_root = oof_path.parent.parent.parent
    return project_root.joinpath(*LEDGER_RELATIVE_PARTS).resolve()


def _canonical_ledger_root(binding: Mapping[str, Any]) -> Path:
    expected = _canonical_ledger_root_from_oof_record(binding.get("oof_ready"))
    embedded = binding.get("canonical_global_ledger_root")
    if not isinstance(embedded, str) or embedded != str(expected):
        raise TestAccessError("M0 scientific binding has a non-canonical global ledger root")
    return expected


def _normalize_run_binding(
    run_root: Path,
    output_paths: Sequence[Path],
    *,
    require_outputs_absent: bool,
) -> dict[str, Any]:
    raw_root = Path(run_root)
    if raw_root.is_symlink():
        raise TestAccessError("run root must be a real non-symlink directory")
    root = raw_root.resolve()
    if not root.is_dir():
        raise TestAccessError("run root must already be a real non-symlink directory")
    if isinstance(output_paths, (str, bytes)) or not isinstance(output_paths, Sequence):
        raise TestAccessError("output_paths must be an explicit non-empty sequence")
    if not output_paths:
        raise TestAccessError("output_paths must be an explicit non-empty sequence")
    normalized: list[str] = []
    for raw_output in output_paths:
        candidate = Path(raw_output)
        if candidate.is_symlink():
            raise TestAccessError("M0 test output path must not be a symlink")
        output = candidate.resolve()
        if output == root or not output.is_relative_to(root):
            raise TestAccessError(f"M0 test output escapes the bound run root: {output}")
        if require_outputs_absent and output.exists():
            raise TestAccessError(f"M0 test output already exists: {output}")
        normalized.append(str(output))
    if len(normalized) != len(set(normalized)):
        raise TestAccessError("M0 test output paths must be unique")
    return {
        "run_root": str(root),
        "output_paths": normalized,
        "output_count": len(normalized),
    }


def _build_binding(
    *,
    experiment_config: Path,
    learning_split: Path,
    identity_manifest: Path,
    oof_ready: Path,
    official_metrics: Path,
    evaluator_script: Path,
    runner_script: Path,
) -> dict[str, Any]:
    config, _, config_record, split_record = _validate_config_and_split(
        experiment_config, learning_split
    )
    _validate_current_experiment(config)
    _, identity_rows, identity_record = _load_identity_manifest(identity_manifest)
    try:
        _, split_receipt = load_and_validate_learning_split(
            Path(learning_split).resolve(), identity_rows, config
        )
    except (RuntimeError, OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TestAccessError(f"learning split is invalid: {exc}") from exc
    case_to_partition = split_receipt.get("case_to_partition")
    if not isinstance(case_to_partition, Mapping):
        raise TestAccessError("validated learning split omits case_to_partition")
    if split_receipt.get("case_counts", {}).get(PARTITION) != EXPECTED_TEST_CASES:
        raise TestAccessError("validated learning split test case count is not 91")
    inventory = _test_inventory(identity_rows, case_to_partition)
    _, oof_record = _validate_oof_identity(oof_ready, identity_rows)

    evaluator_record = _file_record(evaluator_script, label="M0 evaluator script")
    runner_record = _file_record(runner_script, label="M0 test runner script")
    if Path(evaluator_record["path"]).name != EXPECTED_EVALUATOR_NAME:
        raise TestAccessError("M0 evaluator script name is not frozen")
    if Path(runner_record["path"]).name != EXPECTED_RUNNER_NAME:
        raise TestAccessError("M0 test runner script name is not frozen")
    metrics_record = _file_record(official_metrics, label="official metrics")
    binding = {
        "scope": SCOPE,
        "allowed_partition": PARTITION,
        "experiment_config": config_record,
        "learning_split": split_record,
        "identity_manifest": identity_record,
        "oof_ready": oof_record,
        "canonical_global_ledger_root": str(
            _canonical_ledger_root_from_oof_record(oof_record)
        ),
        "official_metrics": metrics_record,
        "m0_evaluator_script": evaluator_record,
        "m0_runner_script": runner_record,
        "test_inventory": inventory,
        "five_fold_ensemble": False,
        "prediction_contract": PREDICTION_CONTRACT,
        "forbidden_downstream_scopes": FORBIDDEN_DOWNSTREAM_SCOPES,
    }
    return {**binding, "binding_sha256": _canonical_sha256(binding)}


def _grant_id(binding: Mapping[str, Any], run_binding: Mapping[str, Any]) -> str:
    return _canonical_sha256(
        {
            "schema_version": GRANT_SCHEMA,
            "binding": binding,
            "run_binding": run_binding,
            "authorization_kind": AUTHORIZATION_KIND,
        }
    )


def _consumption_key(binding: Mapping[str, Any]) -> str:
    # Code, grant, output, and run paths deliberately cannot reset test access.
    return _canonical_sha256(
        {
            "schema_version": LEDGER_SCHEMA,
            "scope": SCOPE,
            "allowed_partition": PARTITION,
            "experiment_config_sha256": binding["experiment_config"]["sha256"],
            "learning_split_sha256": binding["learning_split"]["sha256"],
            "identity_manifest_sha256": binding["identity_manifest"]["sha256"],
            "oof_ready_sha256": binding["oof_ready"]["sha256"],
            "test_case_inventory_sha256": binding["test_inventory"][
                "case_inventory_sha256"
            ],
            "test_patient_inventory_sha256": binding["test_inventory"][
                "patient_inventory_sha256"
            ],
        }
    )


def _validate_embedded_file_record(record: Any, *, label: str) -> dict[str, Any]:
    """Validate a sealed file record without opening the referenced file."""

    if not isinstance(record, Mapping) or set(record) != {"path", "sha256", "bytes"}:
        raise TestAccessError(f"embedded {label} record is invalid")
    path = record.get("path")
    digest = record.get("sha256")
    size = record.get("bytes")
    if (
        not isinstance(path, str)
        or not path
        or not Path(path).is_absolute()
        or not isinstance(digest, str)
        or not HEX_64.fullmatch(digest)
        or isinstance(size, bool)
        or not isinstance(size, int)
        or size <= 0
    ):
        raise TestAccessError(f"embedded {label} record is invalid")
    return dict(record)


def _validate_embedded_test_inventory(value: Any) -> dict[str, Any]:
    """Validate the receipt inventory using only its sealed JSON values."""

    expected_fields = {
        "case_count",
        "patient_count",
        "cases",
        "patients",
        "held_out_fold_case_counts",
        "case_inventory_sha256",
        "patient_inventory_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != expected_fields:
        raise TestAccessError("embedded test inventory contract is invalid")
    cases = value.get("cases")
    patients = value.get("patients")
    if (
        value.get("case_count") != EXPECTED_TEST_CASES
        or value.get("patient_count") != EXPECTED_TEST_PATIENTS
        or not isinstance(cases, list)
        or len(cases) != EXPECTED_TEST_CASES
        or not isinstance(patients, list)
        or len(patients) != EXPECTED_TEST_PATIENTS
        or value.get("held_out_fold_case_counts")
        != EXPECTED_TEST_FOLD_CASE_COUNTS
    ):
        raise TestAccessError("embedded test inventory count is invalid")
    case_map: dict[str, tuple[str, int]] = {}
    for row in cases:
        if not isinstance(row, Mapping) or set(row) != {
            "case_id",
            "patient_id",
            "held_out_fold",
        }:
            raise TestAccessError("embedded test case inventory is invalid")
        case_id = row.get("case_id")
        patient_id = row.get("patient_id")
        fold = row.get("held_out_fold")
        if (
            not isinstance(case_id, str)
            or not case_id
            or case_id in case_map
            or not isinstance(patient_id, str)
            or not patient_id
            or isinstance(fold, bool)
            or not isinstance(fold, int)
            or fold not in range(EXPECTED_FOLDS)
        ):
            raise TestAccessError("embedded test case inventory is invalid")
        case_map[case_id] = (patient_id, fold)
    if list(case_map) != sorted(case_map):
        raise TestAccessError("embedded test case inventory is not sorted")

    patient_map: dict[str, tuple[int, list[str]]] = {}
    for row in patients:
        if not isinstance(row, Mapping) or set(row) != {
            "patient_id",
            "held_out_fold",
            "case_ids",
        }:
            raise TestAccessError("embedded test patient inventory is invalid")
        patient_id = row.get("patient_id")
        fold = row.get("held_out_fold")
        case_ids = row.get("case_ids")
        if (
            not isinstance(patient_id, str)
            or not patient_id
            or patient_id in patient_map
            or isinstance(fold, bool)
            or not isinstance(fold, int)
            or fold not in range(EXPECTED_FOLDS)
            or not isinstance(case_ids, list)
            or not case_ids
            or any(not isinstance(case_id, str) or not case_id for case_id in case_ids)
            or case_ids != sorted(set(case_ids))
        ):
            raise TestAccessError("embedded test patient inventory is invalid")
        patient_map[patient_id] = (fold, list(case_ids))
    if list(patient_map) != sorted(patient_map):
        raise TestAccessError("embedded test patient inventory is not sorted")
    projected = {
        case_id: (patient_id, fold)
        for patient_id, (fold, case_ids) in patient_map.items()
        for case_id in case_ids
    }
    if projected != case_map:
        raise TestAccessError("embedded test case/patient inventories differ")
    if value.get("case_inventory_sha256") != _canonical_sha256(cases):
        raise TestAccessError("embedded test case inventory hash mismatch")
    if value.get("patient_inventory_sha256") != _canonical_sha256(patients):
        raise TestAccessError("embedded test patient inventory hash mismatch")
    return dict(value)


def _validate_embedded_binding(value: Any) -> dict[str, Any]:
    """Validate a receipt binding structurally, without scientific file I/O."""

    file_fields = {
        "experiment_config",
        "learning_split",
        "identity_manifest",
        "oof_ready",
        "official_metrics",
        "m0_evaluator_script",
        "m0_runner_script",
    }
    expected_fields = file_fields | {
        "scope",
        "allowed_partition",
        "test_inventory",
        "five_fold_ensemble",
        "prediction_contract",
        "forbidden_downstream_scopes",
        "binding_sha256",
        "canonical_global_ledger_root",
    }
    if not isinstance(value, Mapping) or set(value) != expected_fields:
        raise TestAccessError("embedded M0 scientific binding contract is invalid")
    binding = dict(value)
    if (
        binding.get("scope") != SCOPE
        or binding.get("allowed_partition") != PARTITION
        or binding.get("five_fold_ensemble") is not False
        or binding.get("prediction_contract") != PREDICTION_CONTRACT
        or binding.get("forbidden_downstream_scopes") != FORBIDDEN_DOWNSTREAM_SCOPES
    ):
        raise TestAccessError("embedded M0 scientific binding scope is invalid")
    for field in file_fields:
        _validate_embedded_file_record(binding.get(field), label=field)
    _canonical_ledger_root(binding)
    _validate_embedded_test_inventory(binding.get("test_inventory"))
    unsigned = {key: item for key, item in binding.items() if key != "binding_sha256"}
    if binding.get("binding_sha256") != _canonical_sha256(unsigned):
        raise TestAccessError("embedded M0 scientific binding hash mismatch")
    return binding


def _validate_embedded_run_binding(value: Any) -> dict[str, Any]:
    """Validate run/output strings without resolving or opening their paths."""

    if not isinstance(value, Mapping) or set(value) != {
        "run_root",
        "output_paths",
        "output_count",
    }:
        raise TestAccessError("embedded M0 run binding contract is invalid")
    root_value = value.get("run_root")
    outputs = value.get("output_paths")
    count = value.get("output_count")
    if (
        not isinstance(root_value, str)
        or not root_value
        or not Path(root_value).is_absolute()
        or not isinstance(outputs, list)
        or not outputs
        or isinstance(count, bool)
        or not isinstance(count, int)
        or count != len(outputs)
        or len(outputs) != len(set(outputs))
    ):
        raise TestAccessError("embedded M0 run binding contract is invalid")
    root = Path(root_value)
    for raw in outputs:
        if not isinstance(raw, str) or not raw:
            raise TestAccessError("embedded M0 output binding is invalid")
        output = Path(raw)
        if (
            not output.is_absolute()
            or ".." in output.parts
            or output == root
            or not output.is_relative_to(root)
        ):
            raise TestAccessError("embedded M0 output binding is invalid")
    return dict(value)


def _validate_grant_envelope(grant_path: Path) -> tuple[Path, dict[str, Any]]:
    """Validate a grant and its raw bindings without reopening bound inputs."""

    grant_path, grant = _load_json(grant_path, label="M0 baseline test grant")
    _verify_seal(grant, "grant_sha256", label="M0 baseline test grant")
    expected_fields = {
        "schema_version",
        "status",
        "scope",
        "allowed_partition",
        "authorized_by",
        "authorization_kind",
        "confirmation",
        "binding",
        "run_binding",
        "grant_id",
        "granted_at_utc",
        "grant_sha256",
    }
    expected = {
        "schema_version": GRANT_SCHEMA,
        "status": "M0_BASELINE_TEST_GRANTED",
        "scope": SCOPE,
        "allowed_partition": PARTITION,
        "authorization_kind": AUTHORIZATION_KIND,
        "confirmation": M0_TEST_CONFIRMATION,
    }
    if set(grant) != expected_fields or any(
        grant.get(key) != expected_value for key, expected_value in expected.items()
    ):
        raise TestAccessError("M0 baseline test grant contract is invalid")
    if grant.get("authorized_by") not in ALLOWED_AUTHORIZERS:
        raise TestAccessError("M0 baseline test grant authorizer is invalid")
    if not isinstance(grant.get("granted_at_utc"), str) or not grant["granted_at_utc"]:
        raise TestAccessError("M0 baseline test grant timestamp is invalid")
    binding = _validate_embedded_binding(grant.get("binding"))
    run_binding = _validate_embedded_run_binding(grant.get("run_binding"))
    if grant.get("grant_id") != _grant_id(binding, run_binding):
        raise TestAccessError("M0 baseline test grant id mismatch")
    return grant_path, grant


def create_m0_test_grant(
    *,
    experiment_config: Path,
    learning_split: Path,
    identity_manifest: Path,
    oof_ready: Path,
    official_metrics: Path,
    evaluator_script: Path,
    runner_script: Path,
    run_root: Path,
    output_paths: Sequence[Path],
    grant_path: Path,
    authorized_by: str,
    confirmation: str,
    now: Callable[[], str] = _utc_now,
) -> dict[str, Any]:
    """Publish one explicit, M0-only grant without consuming it."""

    if authorized_by not in ALLOWED_AUTHORIZERS:
        raise TestAccessError("authorized_by must identify the director or delegated Codex")
    if confirmation != M0_TEST_CONFIRMATION:
        raise TestAccessError("exact M0 baseline test confirmation phrase is required")
    binding = _build_binding(
        experiment_config=experiment_config,
        learning_split=learning_split,
        identity_manifest=identity_manifest,
        oof_ready=oof_ready,
        official_metrics=official_metrics,
        evaluator_script=evaluator_script,
        runner_script=runner_script,
    )
    run_binding = _normalize_run_binding(
        run_root, output_paths, require_outputs_absent=True
    )
    payload = _sealed(
        {
            "schema_version": GRANT_SCHEMA,
            "status": "M0_BASELINE_TEST_GRANTED",
            "scope": SCOPE,
            "allowed_partition": PARTITION,
            "authorized_by": authorized_by,
            "authorization_kind": AUTHORIZATION_KIND,
            "confirmation": confirmation,
            "binding": binding,
            "run_binding": run_binding,
            "grant_id": _grant_id(binding, run_binding),
            "granted_at_utc": now(),
        },
        "grant_sha256",
    )
    try:
        _write_json_exclusive(grant_path, payload)
    except FileExistsError as exc:
        raise TestAccessError(f"M0 baseline test grant already exists: {grant_path}") from exc
    return payload


def _validate_grant(
    grant_path: Path, *, require_outputs_absent: bool
) -> tuple[Path, dict[str, Any]]:
    grant_path, grant = _validate_grant_envelope(grant_path)
    binding = grant.get("binding")
    run_binding = grant.get("run_binding")
    current = _build_binding(
        experiment_config=_required_record_path(
            binding.get("experiment_config"), label="experiment config"
        ),
        learning_split=_required_record_path(
            binding.get("learning_split"), label="learning split"
        ),
        identity_manifest=_required_record_path(
            binding.get("identity_manifest"), label="identity manifest"
        ),
        oof_ready=_required_record_path(binding.get("oof_ready"), label="OOF_READY"),
        official_metrics=_required_record_path(
            binding.get("official_metrics"), label="official metrics"
        ),
        evaluator_script=_required_record_path(
            binding.get("m0_evaluator_script"), label="M0 evaluator script"
        ),
        runner_script=_required_record_path(
            binding.get("m0_runner_script"), label="M0 runner script"
        ),
    )
    if current != binding:
        raise TestAccessError("M0 baseline test scientific/code binding changed after grant")
    output_paths = run_binding.get("output_paths")
    if not isinstance(output_paths, list) or not all(
        isinstance(path, str) and path for path in output_paths
    ):
        raise TestAccessError("M0 baseline test grant output binding is invalid")
    current_run = _normalize_run_binding(
        Path(str(run_binding.get("run_root") or "")),
        tuple(Path(path) for path in output_paths),
        require_outputs_absent=require_outputs_absent,
    )
    if current_run != run_binding:
        raise TestAccessError("M0 baseline test run/output binding changed after grant")
    if grant.get("grant_id") != _grant_id(binding, run_binding):
        raise TestAccessError("M0 baseline test grant id mismatch")
    return grant_path, grant


def consume_m0_test_grant(
    *,
    grant_path: Path,
    run_root: Path,
    output_paths: Sequence[Path],
    receipt_path: Path,
    ledger_root: Path = GLOBAL_LEDGER_ROOT,
    now: Callable[[], str] = _utc_now,
) -> dict[str, Any]:
    """Atomically consume the global M0-only test decision."""

    grant_path, grant = _validate_grant(
        grant_path, require_outputs_absent=True
    )
    requested_run = _normalize_run_binding(
        run_root, output_paths, require_outputs_absent=True
    )
    if requested_run != grant["run_binding"]:
        raise TestAccessError("M0 test consumption differs from the granted run/outputs")
    root = Path(requested_run["run_root"])
    receipt_path = Path(receipt_path).resolve()
    if receipt_path == root or not receipt_path.is_relative_to(root):
        raise TestAccessError("M0 consumed receipt must be inside its bound run root")
    if str(receipt_path) in requested_run["output_paths"]:
        raise TestAccessError("M0 consumed receipt cannot replace a bound scientific output")

    binding = dict(grant["binding"])
    raw_ledger_root = Path(ledger_root)
    if raw_ledger_root.is_symlink():
        raise TestAccessError("M0 global ledger root must not be a symlink")
    ledger_root = raw_ledger_root.resolve()
    canonical_ledger_root = _canonical_ledger_root(binding)
    if ledger_root != canonical_ledger_root:
        raise TestAccessError("M0 global ledger root is not the canonical project ledger")
    if ledger_root == root or ledger_root.is_relative_to(root):
        raise TestAccessError("M0 global ledger must be independent of the run root")
    key = _consumption_key(binding)
    ledger_path = ledger_root / f"{key}.json"
    core = {
        "consumption_key": key,
        "grant_id": grant["grant_id"],
        "grant": _file_record(grant_path, label="M0 baseline test grant"),
        "scope": SCOPE,
        "allowed_partition": PARTITION,
        "authorization_kind": AUTHORIZATION_KIND,
        "authorized_by": grant["authorized_by"],
        "binding": binding,
        "run_binding": dict(grant["run_binding"]),
        "receipt_path": str(receipt_path),
        "consumed_at_utc": now(),
    }
    core_sha256 = _canonical_sha256(core)
    ledger = _sealed(
        {
            "schema_version": LEDGER_SCHEMA,
            "status": "CONSUMED",
            "scope": SCOPE,
            "allowed_partition": PARTITION,
            "consumption": core,
            "consumption_sha256": core_sha256,
        },
        "ledger_record_sha256",
    )
    try:
        _write_json_exclusive(ledger_path, ledger)
    except FileExistsError as exc:
        raise TestAccessError(
            "M0 baseline test access was already consumed globally"
        ) from exc
    receipt = _sealed(
        {
            "schema_version": RECEIPT_SCHEMA,
            "status": "CONSUMED",
            "scope": SCOPE,
            "allowed_partition": PARTITION,
            "consumption": core,
            "consumption_sha256": core_sha256,
            "global_ledger": _file_record(
                ledger_path, label="M0 global consumption ledger record"
            ),
        },
        "receipt_sha256",
    )
    try:
        _write_json_exclusive(receipt_path, receipt)
    except Exception as exc:
        # The global claim remains intentionally consumed after partial failure.
        raise TestAccessError(
            "M0 test access was consumed but its run receipt was not published"
        ) from exc
    return receipt


def validate_m0_test_receipt(
    receipt_path: Path,
    *,
    experiment_config: Path,
    learning_split: Path,
    identity_manifest: Path,
    oof_ready: Path,
    official_metrics: Path,
    evaluator_script: Path,
    runner_script: Path,
    run_root: Path,
    output_paths: Sequence[Path],
    ledger_root: Path = GLOBAL_LEDGER_ROOT,
) -> dict[str, Any]:
    """Revalidate the M0 receipt, all inputs, and its global ledger anchor."""

    receipt_path, receipt = _load_json(receipt_path, label="M0 consumed test receipt")
    _verify_seal(receipt, "receipt_sha256", label="M0 consumed test receipt")
    expected_receipt_fields = {
        "schema_version",
        "status",
        "scope",
        "allowed_partition",
        "consumption",
        "consumption_sha256",
        "global_ledger",
        "receipt_sha256",
    }
    expected = {
        "schema_version": RECEIPT_SCHEMA,
        "status": "CONSUMED",
        "scope": SCOPE,
        "allowed_partition": PARTITION,
    }
    if set(receipt) != expected_receipt_fields or any(
        receipt.get(key) != value for key, value in expected.items()
    ):
        raise TestAccessError("M0 consumed test receipt contract is invalid")
    core = receipt.get("consumption")
    expected_core_fields = {
        "consumption_key",
        "grant_id",
        "grant",
        "scope",
        "allowed_partition",
        "authorization_kind",
        "authorized_by",
        "binding",
        "run_binding",
        "receipt_path",
        "consumed_at_utc",
    }
    if not isinstance(core, Mapping) or set(core) != expected_core_fields:
        raise TestAccessError("M0 consumed test receipt omits its consumption core")
    if receipt.get("consumption_sha256") != _canonical_sha256(core):
        raise TestAccessError("M0 consumed test receipt core hash mismatch")
    if (
        core.get("scope") != SCOPE
        or core.get("allowed_partition") != PARTITION
        or core.get("authorization_kind") != AUTHORIZATION_KIND
        or core.get("authorized_by") not in ALLOWED_AUTHORIZERS
    ):
        raise TestAccessError("M0 consumed test receipt is not test-only baseline scope")
    if (
        not isinstance(core.get("consumed_at_utc"), str)
        or not core["consumed_at_utc"]
        or core.get("receipt_path") != str(receipt_path)
    ):
        raise TestAccessError("M0 consumed receipt path differs from its global claim")

    # Establish the independent authorization anchors using only values sealed
    # inside the receipt.  Scientific inputs are deliberately not opened until
    # both the project-global O_EXCL ledger and the original grant have passed.
    embedded_binding = _validate_embedded_binding(core.get("binding"))
    embedded_run = _validate_embedded_run_binding(core.get("run_binding"))
    expected_key = _consumption_key(embedded_binding)
    if core.get("consumption_key") != expected_key:
        raise TestAccessError("M0 consumed receipt global key mismatch")

    raw_ledger_root = Path(ledger_root)
    if raw_ledger_root.is_symlink():
        raise TestAccessError("M0 global ledger root must not be a symlink")
    ledger_root = raw_ledger_root.resolve()
    canonical_ledger_root = _canonical_ledger_root(embedded_binding)
    if ledger_root != canonical_ledger_root:
        raise TestAccessError("M0 global ledger root is not the canonical project ledger")
    embedded_root = Path(embedded_run["run_root"]).resolve()
    if ledger_root == embedded_root or ledger_root.is_relative_to(embedded_root):
        raise TestAccessError("M0 global ledger must be independent of the run root")
    ledger_path = ledger_root / f"{expected_key}.json"
    ledger_record = _validate_embedded_file_record(
        receipt.get("global_ledger"), label="M0 global consumption ledger record"
    )
    if ledger_record.get("path") != str(ledger_path) or ledger_record != _file_record(
        ledger_path, label="M0 global consumption ledger record"
    ):
        raise TestAccessError("M0 consumed receipt global ledger record changed")
    _, ledger = _load_json(ledger_path, label="M0 global consumption ledger record")
    _verify_seal(
        ledger,
        "ledger_record_sha256",
        label="M0 global consumption ledger record",
    )
    if (
        ledger.get("schema_version") != LEDGER_SCHEMA
        or ledger.get("status") != "CONSUMED"
        or ledger.get("scope") != SCOPE
        or ledger.get("allowed_partition") != PARTITION
        or ledger.get("consumption") != core
        or ledger.get("consumption_sha256") != receipt["consumption_sha256"]
    ):
        raise TestAccessError("M0 global consumption ledger differs from the run receipt")

    grant_record = _validate_embedded_file_record(
        core.get("grant"), label="M0 baseline test grant"
    )
    grant_path = _required_record_path(grant_record, label="M0 baseline test grant")
    if grant_record != _file_record(
        grant_path,
        label="M0 baseline test grant",
    ):
        raise TestAccessError("M0 baseline test grant changed after consumption")
    _, grant = _validate_grant_envelope(grant_path)
    if (
        grant.get("grant_id") != core.get("grant_id")
        or grant.get("binding") != embedded_binding
        or grant.get("run_binding") != embedded_run
        or grant.get("authorized_by") != core.get("authorized_by")
    ):
        raise TestAccessError("M0 consumed receipt differs from its grant")

    # Authorization is now independently anchored.  Only at this point may the
    # validator read identity/split/OOF/code inputs and compare their live bytes
    # with the sealed scientific binding.
    current_binding = _build_binding(
        experiment_config=experiment_config,
        learning_split=learning_split,
        identity_manifest=identity_manifest,
        oof_ready=oof_ready,
        official_metrics=official_metrics,
        evaluator_script=evaluator_script,
        runner_script=runner_script,
    )
    if embedded_binding != current_binding:
        raise TestAccessError("M0 consumed receipt scientific/code binding changed")
    current_run = _normalize_run_binding(
        run_root, output_paths, require_outputs_absent=False
    )
    if embedded_run != current_run:
        raise TestAccessError("M0 consumed receipt run/output binding differs")
    receipt_hash = receipt.get("receipt_sha256")
    if not isinstance(receipt_hash, str) or not HEX_64.fullmatch(receipt_hash):
        raise TestAccessError("M0 consumed receipt_sha256 is invalid")
    return dict(receipt)


def enforce_m0_test_access(
    *,
    receipt_path: Path,
    experiment_config: Path,
    learning_split: Path,
    identity_manifest: Path,
    oof_ready: Path,
    official_metrics: Path,
    evaluator_script: Path,
    runner_script: Path,
    run_root: Path,
    output_paths: Sequence[Path],
    ledger_root: Path = GLOBAL_LEDGER_ROOT,
) -> dict[str, Any]:
    """Fail closed unless this is the exact consumed M0 test-only run."""

    return validate_m0_test_receipt(
        receipt_path,
        experiment_config=experiment_config,
        learning_split=learning_split,
        identity_manifest=identity_manifest,
        oof_ready=oof_ready,
        official_metrics=official_metrics,
        evaluator_script=evaluator_script,
        runner_script=runner_script,
        run_root=run_root,
        output_paths=output_paths,
        ledger_root=ledger_root,
    )


def _add_binding_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--experiment-config", type=Path, required=True)
    parser.add_argument("--learning-split", type=Path, required=True)
    parser.add_argument("--identity-manifest", type=Path, required=True)
    parser.add_argument("--oof-ready", type=Path, required=True)
    parser.add_argument("--official-metrics", type=Path, required=True)
    parser.add_argument("--evaluator-script", type=Path, required=True)
    parser.add_argument("--runner-script", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, action="append", required=True)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    grant = commands.add_parser("grant", help="publish the explicit M0-only grant")
    _add_binding_arguments(grant)
    grant.add_argument("--grant", type=Path, required=True)
    grant.add_argument("--authorized-by", choices=ALLOWED_AUTHORIZERS, required=True)
    grant.add_argument("--confirm-m0-test", required=True)
    consume = commands.add_parser("consume", help="atomically consume the M0 grant")
    consume.add_argument("--grant", type=Path, required=True)
    consume.add_argument("--run-root", type=Path, required=True)
    consume.add_argument("--output-path", type=Path, action="append", required=True)
    consume.add_argument("--receipt", type=Path, required=True)
    consume.add_argument("--ledger-root", type=Path, required=True)
    validate = commands.add_parser("validate", help="revalidate an M0 receipt")
    _add_binding_arguments(validate)
    validate.add_argument("--receipt", type=Path, required=True)
    validate.add_argument("--ledger-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "grant":
            payload = create_m0_test_grant(
                experiment_config=args.experiment_config,
                learning_split=args.learning_split,
                identity_manifest=args.identity_manifest,
                oof_ready=args.oof_ready,
                official_metrics=args.official_metrics,
                evaluator_script=args.evaluator_script,
                runner_script=args.runner_script,
                run_root=args.run_root,
                output_paths=args.output_path,
                grant_path=args.grant,
                authorized_by=args.authorized_by,
                confirmation=args.confirm_m0_test,
            )
        elif args.command == "consume":
            payload = consume_m0_test_grant(
                grant_path=args.grant,
                run_root=args.run_root,
                output_paths=args.output_path,
                receipt_path=args.receipt,
                ledger_root=args.ledger_root,
            )
        else:
            payload = validate_m0_test_receipt(
                args.receipt,
                experiment_config=args.experiment_config,
                learning_split=args.learning_split,
                identity_manifest=args.identity_manifest,
                oof_ready=args.oof_ready,
                official_metrics=args.official_metrics,
                evaluator_script=args.evaluator_script,
                runner_script=args.runner_script,
                run_root=args.run_root,
                output_paths=args.output_path,
                ledger_root=args.ledger_root,
            )
    except TestAccessError as exc:
        parser.error(str(exc))
    sys.stdout.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
