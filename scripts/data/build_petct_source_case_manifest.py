#!/usr/bin/env python3
"""Build the two-stage, truth-firewalled 597-case PSMA source manifest.

``identity`` inventories case/patient/fold/path identity without opening a NIfTI
or hashing a source leaf.  After the patient learning split is frozen,
``materialize`` opens and hashes only explicitly authorized partitions while
retaining locked identity rows for the rest of the 597-case cohort.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import nibabel as nib
import numpy as np


SCRIPTS = Path(__file__).resolve().parents[1]
for directory in (SCRIPTS, SCRIPTS / "common", SCRIPTS / "data"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from common.petct_learning import sha256_file  # noqa: E402
from common.petct_test_access import (  # noqa: E402
    TestAccessError,
    add_leaf_test_access_arguments,
    enforce_partition_access,
)
from data.audit_psma_v3_dataset import AUDIT_VERSION, discover_triplets  # noqa: E402
from data.validate_petct_learning_split import (  # noqa: E402
    PARTITION_SET,
    load_and_validate_learning_split,
)


EXPECTED_CASES = 597
EXPECTED_PATIENTS = 378
EXPECTED_FOLDS = 5
RECEIPT_SCHEMA = "PETCT-SOURCE-CASE-MANIFEST-RECEIPT-v2.0"
IDENTITY_MODE = "identity"
MATERIALIZE_MODE = "materialize"
IDENTITY_STATE = "IDENTITY_ONLY"
MATERIALIZED_STATE = "MATERIALIZED_AUTHORIZED"
LOCKED_STATE = "LOCKED_UNREAD"


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _regular(path: Path, *, label: str) -> Path:
    if path.is_symlink():
        raise RuntimeError(f"missing regular {label}: {path}")
    resolved = path.resolve()
    if not resolved.is_file():
        raise RuntimeError(f"missing regular {label}: {resolved}")
    return resolved


def _load_json(path: Path, *, label: str) -> Any:
    path = _regular(path, label=label)
    try:
        with path.open("r", encoding="utf-8") as stream:
            return json.load(stream)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid {label} JSON") from exc


def _validate_audit(
    audit: Mapping[str, Any],
    *,
    dataset_root: Path,
    splits_sha256: str,
    expected_cases: int,
    expected_patients: int,
) -> tuple[dict[str, Mapping[str, Any]], dict[str, str]]:
    if not isinstance(audit, Mapping):
        raise RuntimeError("audit must be an object")
    if audit.get("status") != "PASS" or audit.get("audit_version") != AUDIT_VERSION:
        raise RuntimeError("source audit is not a complete PASS for the expected version")
    if audit.get("errors") not in ([], None):
        raise RuntimeError("PASS audit unexpectedly contains errors")
    if Path(str(audit.get("dataset_root") or "")).resolve() != dataset_root:
        raise RuntimeError("audit dataset_root differs from requested dataset root")
    source_hashes = audit.get("source_hashes")
    if not isinstance(source_hashes, Mapping):
        raise RuntimeError("audit source_hashes are missing")
    if source_hashes.get("splits_final_sha256") != splits_sha256:
        raise RuntimeError("frozen splits hash differs from the PASS audit")
    summary = audit.get("summary")
    if not isinstance(summary, Mapping):
        raise RuntimeError("audit summary is missing")
    expected_summary = {
        "case_count": expected_cases,
        "patient_count": expected_patients,
        "failed_case_count": 0,
        "unreadable_label_count": 0,
        "invalid_label_count": 0,
    }
    for key, expected in expected_summary.items():
        if summary.get(key) != expected:
            raise RuntimeError(f"audit summary {key} differs from frozen expectation")
    per_case = audit.get("per_case")
    if not isinstance(per_case, list) or len(per_case) != expected_cases:
        raise RuntimeError("audit per_case does not contain the complete case inventory")
    rows: dict[str, Mapping[str, Any]] = {}
    patient_by_case: dict[str, str] = {}
    for index, row in enumerate(per_case, start=1):
        if not isinstance(row, Mapping):
            raise RuntimeError(f"audit per_case row {index} must be an object")
        case_id = str(row.get("case_id") or "")
        patient_id = str(row.get("patient_id") or "").casefold()
        shape = row.get("shape")
        if not case_id or case_id in rows or not patient_id:
            raise RuntimeError("audit contains missing/duplicate case or patient identity")
        if row.get("status") != "PASS" or row.get("errors") not in ([], None):
            raise RuntimeError(f"audit case is not PASS: {case_id}")
        if (
            not isinstance(shape, list)
            or len(shape) != 3
            or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in shape)
        ):
            raise RuntimeError(f"audit case has invalid shape: {case_id}")
        rows[case_id] = row
        patient_by_case[case_id] = patient_id
    if len(set(patient_by_case.values())) != expected_patients:
        raise RuntimeError("audit per_case patient count differs from frozen expectation")
    return rows, patient_by_case


def _validate_splits(
    splits: Any,
    *,
    cases: set[str],
    patient_by_case: Mapping[str, str],
    expected_folds: int,
) -> dict[str, int]:
    if not isinstance(splits, list) or len(splits) != expected_folds:
        raise RuntimeError("splits_final must contain exactly the frozen number of folds")
    validation_counts: Counter[str] = Counter()
    patient_folds: dict[str, set[int]] = defaultdict(set)
    held_out_fold: dict[str, int] = {}
    for fold_index, fold in enumerate(splits):
        if not isinstance(fold, Mapping):
            raise RuntimeError(f"split fold {fold_index} must be an object")
        train_raw, val_raw = fold.get("train"), fold.get("val")
        if not isinstance(train_raw, list) or not isinstance(val_raw, list):
            raise RuntimeError(f"split fold {fold_index} requires train and val lists")
        train = [str(case_id) for case_id in train_raw]
        val = [str(case_id) for case_id in val_raw]
        if len(train) != len(set(train)) or len(val) != len(set(val)):
            raise RuntimeError(f"split fold {fold_index} contains duplicate case ids")
        train_set, val_set = set(train), set(val)
        if train_set & val_set:
            raise RuntimeError(f"split fold {fold_index} has train/val case overlap")
        if train_set | val_set != cases or train_set != cases - val_set:
            raise RuntimeError(f"split fold {fold_index} is not the full val complement")
        train_patients = {patient_by_case[case_id] for case_id in train_set}
        val_patients = {patient_by_case[case_id] for case_id in val_set}
        if train_patients & val_patients:
            raise RuntimeError(f"split fold {fold_index} has patient leakage")
        for case_id in val:
            validation_counts[case_id] += 1
            held_out_fold[case_id] = fold_index
            patient_folds[patient_by_case[case_id]].add(fold_index)
    not_once = {case_id: validation_counts[case_id] for case_id in cases if validation_counts[case_id] != 1}
    if not_once:
        raise RuntimeError("every case must appear in exactly one validation fold")
    if any(len(folds) != 1 for folds in patient_folds.values()):
        raise RuntimeError("one patient appears in multiple validation folds")
    if set(held_out_fold) != cases:
        raise RuntimeError("held-out fold mapping does not cover the source inventory")
    return held_out_fold


def build_identity_case_rows(
    dataset_root: Path,
    splits: Any,
    audit: Mapping[str, Any],
    *,
    splits_sha256: str,
    expected_cases: int = EXPECTED_CASES,
    expected_patients: int = EXPECTED_PATIENTS,
    expected_folds: int = EXPECTED_FOLDS,
) -> tuple[list[dict[str, Any]], str]:
    """Return complete path identity without opening or hashing any NIfTI leaf."""

    dataset_root = dataset_root.resolve()
    if dataset_root.is_symlink() or not dataset_root.is_dir():
        raise RuntimeError(f"dataset root is missing or a symlink: {dataset_root}")
    audit_rows, patient_by_case = _validate_audit(
        audit,
        dataset_root=dataset_root,
        splits_sha256=splits_sha256,
        expected_cases=expected_cases,
        expected_patients=expected_patients,
    )
    triplets, path_errors = discover_triplets(dataset_root)
    if path_errors:
        raise RuntimeError("dataset path inventory is invalid: " + ";".join(path_errors))
    if len(triplets) != expected_cases or set(triplets) != set(audit_rows):
        raise RuntimeError("path inventory differs from the complete PASS audit")
    held_out_fold = _validate_splits(
        splits,
        cases=set(triplets),
        patient_by_case=patient_by_case,
        expected_folds=expected_folds,
    )

    rows: list[dict[str, Any]] = []
    for case_id in sorted(triplets):
        # ``discover_triplets`` classifies names only.  Do not resolve/check/stat
        # the leaf here: an unreadable locked test label must not affect identity.
        ct_path, pet_path, gt_path = triplets[case_id]
        row = {
            "case_id": case_id,
            "patient_id": patient_by_case[case_id],
            "held_out_fold": held_out_fold[case_id],
            "ct_path": str(Path(os.path.abspath(ct_path))),
            "pet_path": str(Path(os.path.abspath(pet_path))),
            "gt_path": str(Path(os.path.abspath(gt_path))),
            "truth_materialization": IDENTITY_STATE,
        }
        rows.append(row)
    if len(rows) != expected_cases or len({row["patient_id"] for row in rows}) != expected_patients:
        raise RuntimeError("generated source manifest count differs from frozen cohort")
    return rows, _canonical_sha256(rows)


def _validate_identity_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_cases: int,
    expected_patients: int,
    expected_folds: int,
) -> list[dict[str, Any]]:
    expected_keys = {
        "case_id",
        "patient_id",
        "held_out_fold",
        "ct_path",
        "pet_path",
        "gt_path",
        "truth_materialization",
    }
    validated: list[dict[str, Any]] = []
    cases: set[str] = set()
    patient_fold: dict[str, int] = {}
    folds: set[int] = set()
    for index, raw in enumerate(rows, start=1):
        if not isinstance(raw, Mapping) or set(raw) != expected_keys:
            raise RuntimeError(f"identity row {index} has an invalid field set")
        row = dict(raw)
        case_id = str(row["case_id"] or "")
        patient_id = str(row["patient_id"] or "").casefold()
        fold = row["held_out_fold"]
        if (
            not case_id
            or case_id in cases
            or not patient_id
            or isinstance(fold, bool)
            or not isinstance(fold, int)
            or fold not in range(expected_folds)
            or row["truth_materialization"] != IDENTITY_STATE
        ):
            raise RuntimeError(f"identity row {index} has invalid identity")
        if patient_id in patient_fold and patient_fold[patient_id] != fold:
            raise RuntimeError("identity manifest has patient fold leakage")
        for field in ("ct_path", "pet_path", "gt_path"):
            value = str(row[field] or "")
            if not value or not Path(value).is_absolute():
                raise RuntimeError(f"identity row {index} requires absolute {field}")
            row[field] = value
        row["patient_id"] = patient_id
        cases.add(case_id)
        folds.add(fold)
        patient_fold[patient_id] = fold
        validated.append(row)
    if len(cases) != expected_cases or len(patient_fold) != expected_patients:
        raise RuntimeError("identity manifest count differs from frozen cohort")
    if folds != set(range(expected_folds)):
        raise RuntimeError("identity manifest does not represent every held-out fold")
    if [row["case_id"] for row in validated] != sorted(cases):
        raise RuntimeError("identity manifest must be sorted by case_id")
    return validated


def _case_partitions_from_split(
    learning_split: Mapping[str, Any], identity_rows: Sequence[Mapping[str, Any]]
) -> dict[str, str]:
    if not isinstance(learning_split, Mapping):
        raise RuntimeError("learning split must be an object")
    patients = learning_split.get("patients")
    if (
        learning_split.get("status") != "FROZEN_BEFORE_MODEL_SELECTION"
        or not isinstance(patients, list)
    ):
        raise RuntimeError("learning split is not frozen or lacks patient rows")
    identity = {str(row["case_id"]): str(row["patient_id"]).casefold() for row in identity_rows}
    assigned: dict[str, str] = {}
    for index, raw in enumerate(patients, start=1):
        if not isinstance(raw, Mapping):
            raise RuntimeError(f"learning split patient row {index} must be an object")
        patient = str(raw.get("patient_id") or "").casefold()
        partition = str(raw.get("partition") or "")
        case_ids = raw.get("case_ids")
        if not patient or partition not in PARTITION_SET or not isinstance(case_ids, list):
            raise RuntimeError(f"learning split patient row {index} is invalid")
        for raw_case_id in case_ids:
            case_id = str(raw_case_id)
            if case_id in assigned or identity.get(case_id) != patient:
                raise RuntimeError("learning split differs from identity case/patient mapping")
            assigned[case_id] = partition
    if set(assigned) != set(identity):
        raise RuntimeError("learning split does not cover the complete identity inventory")
    return assigned


def materialize_source_case_rows(
    identity_rows: Sequence[Mapping[str, Any]],
    learning_split: Mapping[str, Any],
    *,
    authorized_partitions: Sequence[str],
    expected_cases: int = EXPECTED_CASES,
    expected_patients: int = EXPECTED_PATIENTS,
    expected_folds: int = EXPECTED_FOLDS,
) -> tuple[list[dict[str, Any]], str]:
    """Hash only authorized leaves; keep every other case explicitly locked."""

    authorized = {str(value) for value in authorized_partitions}
    if not authorized or not authorized.issubset(PARTITION_SET):
        raise RuntimeError("authorized partitions must be an explicit subset of train/val/test")
    identity = _validate_identity_rows(
        identity_rows,
        expected_cases=expected_cases,
        expected_patients=expected_patients,
        expected_folds=expected_folds,
    )
    case_partition = _case_partitions_from_split(learning_split, identity)
    rows: list[dict[str, Any]] = []
    for source in identity:
        row = {key: value for key, value in source.items() if key != "truth_materialization"}
        case_id = str(row["case_id"])
        partition = case_partition[case_id]
        row["partition"] = partition
        if partition not in authorized:
            row["truth_materialization"] = LOCKED_STATE
            rows.append(row)
            continue

        paths = [
            _regular(Path(str(row[f"{modality}_path"])), label=f"{case_id} {modality}")
            for modality in ("ct", "pet", "gt")
        ]
        images = [nib.load(str(path)) for path in paths]
        shapes = [tuple(int(value) for value in image.shape[:3]) for image in images]
        if len(set(shapes)) != 1:
            raise RuntimeError(f"selected CT/PET/GT shape drift: {case_id}")
        reference_affine = np.asarray(images[0].affine)
        if any(
            not np.allclose(reference_affine, np.asarray(image.affine), rtol=0.0, atol=1e-4)
            for image in images[1:]
        ):
            raise RuntimeError(f"selected CT/PET/GT affine drift: {case_id}")
        for modality, path in zip(("ct", "pet", "gt"), paths):
            row[f"{modality}_path"] = str(path)
            row[f"{modality}_bytes"] = path.stat().st_size
            row[f"{modality}_sha256"] = sha256_file(path)
        row["nifti_shape"] = list(shapes[0])
        row["truth_materialization"] = MATERIALIZED_STATE
        rows.append(row)
    return rows, _canonical_sha256(rows)


def validate_materialized_source_case_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    authorized_partitions: Sequence[str],
    expected_cases: int = EXPECTED_CASES,
    expected_patients: int = EXPECTED_PATIENTS,
    expected_folds: int = EXPECTED_FOLDS,
) -> dict[str, Any]:
    """Revalidate selected leaf bytes without ever opening locked partitions."""

    authorized = {str(value) for value in authorized_partitions}
    if not authorized or not authorized.issubset(PARTITION_SET):
        raise RuntimeError("authorized partitions must be an explicit subset of train/val/test")
    identity_keys = {
        "case_id",
        "patient_id",
        "held_out_fold",
        "ct_path",
        "pet_path",
        "gt_path",
    }
    identity_projection = [
        {
            **{key: raw[key] for key in identity_keys},
            "truth_materialization": IDENTITY_STATE,
        }
        for raw in rows
    ]
    _validate_identity_rows(
        identity_projection,
        expected_cases=expected_cases,
        expected_patients=expected_patients,
        expected_folds=expected_folds,
    )
    materialized = 0
    locked = 0
    for index, raw in enumerate(rows, start=1):
        partition = str(raw.get("partition") or "")
        if partition not in PARTITION_SET:
            raise RuntimeError(f"source row {index} has invalid partition")
        base_keys = identity_keys | {"partition", "truth_materialization"}
        if partition not in authorized:
            if set(raw) != base_keys or raw.get("truth_materialization") != LOCKED_STATE:
                raise RuntimeError("locked source row exposes materialized truth")
            locked += 1
            continue
        expected_keys = base_keys | {
            "nifti_shape",
            *(f"{modality}_{suffix}" for modality in ("ct", "pet", "gt") for suffix in ("bytes", "sha256")),
        }
        if set(raw) != expected_keys or raw.get("truth_materialization") != MATERIALIZED_STATE:
            raise RuntimeError("authorized source row lacks exact materialized leaf contract")
        for modality in ("ct", "pet", "gt"):
            path = _regular(Path(str(raw[f"{modality}_path"])), label=f"selected {modality}")
            if path.stat().st_size != raw[f"{modality}_bytes"]:
                raise RuntimeError(f"selected {modality.upper()} byte-size drift")
            if sha256_file(path) != raw[f"{modality}_sha256"]:
                raise RuntimeError(f"selected {modality.upper()} content hash drift")
        materialized += 1
    return {
        "status": "PASS",
        "case_count": len(rows),
        "materialized_case_count": materialized,
        "locked_unread_case_count": locked,
        "authorized_partitions": sorted(authorized),
    }


def publish_jsonl(rows: Sequence[Mapping[str, Any]], output: Path) -> str:
    output = output.resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite source case manifest: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    return sha256_file(output)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=(IDENTITY_MODE, MATERIALIZE_MODE), required=True)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--splits-final", type=Path)
    parser.add_argument("--audit-json", type=Path)
    parser.add_argument("--identity-manifest", type=Path)
    parser.add_argument("--learning-split", type=Path)
    parser.add_argument("--experiment-config", type=Path)
    parser.add_argument("--partitions", nargs="+", choices=sorted(PARTITION_SET))
    parser.add_argument("--output", type=Path, required=True)
    add_leaf_test_access_arguments(parser)
    args = parser.parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        parser.error("output already exists")

    input_sha256: dict[str, str]
    if args.mode == IDENTITY_MODE:
        if any(value is None for value in (args.dataset_root, args.splits_final, args.audit_json)):
            parser.error("identity mode requires --dataset-root, --splits-final and --audit-json")
        if any(
            value is not None
            for value in (
                args.identity_manifest,
                args.learning_split,
                args.experiment_config,
                args.partitions,
                args.test_access_receipt,
                args.run_root,
            )
        ):
            parser.error("identity mode rejects materialization and test-access arguments")
        splits_path = _regular(args.splits_final, label="splits_final")
        audit_path = _regular(args.audit_json, label="PASS audit")
        splits_sha = sha256_file(splits_path)
        rows, inventory_sha = build_identity_case_rows(
            args.dataset_root,
            _load_json(splits_path, label="splits_final"),
            _load_json(audit_path, label="PASS audit"),
            splits_sha256=splits_sha,
        )
        input_sha256 = {
            "splits_final": splits_sha,
            "audit_json": sha256_file(audit_path),
            "identity_inventory": inventory_sha,
        }
    else:
        if any(
            value is None
            for value in (
                args.identity_manifest,
                args.learning_split,
                args.experiment_config,
                args.partitions,
            )
        ):
            parser.error(
                "materialize mode requires --identity-manifest, --learning-split, "
                "--experiment-config and explicit --partitions"
            )
        if any(value is not None for value in (args.dataset_root, args.splits_final, args.audit_json)):
            parser.error("materialize mode rejects dataset discovery inputs")
        try:
            access_receipt = enforce_partition_access(
                args.partitions,
                receipt_path=args.test_access_receipt,
                experiment_config=args.experiment_config,
                learning_split=args.learning_split,
                run_root=args.run_root,
                output_paths=(args.output,),
            )
        except TestAccessError as exc:
            parser.error(str(exc))
        identity_path = _regular(args.identity_manifest, label="identity manifest")
        learning_split_path = _regular(args.learning_split, label="learning split")
        experiment_config_path = _regular(args.experiment_config, label="experiment config")
        from common.petct_learning import load_jsonl  # noqa: PLC0415

        identity_rows = load_jsonl(identity_path)
        experiment_config = _load_json(experiment_config_path, label="experiment config")
        learning_split, validated_split = load_and_validate_learning_split(
            learning_split_path, identity_rows, experiment_config
        )
        rows, inventory_sha = materialize_source_case_rows(
            identity_rows,
            learning_split,
            authorized_partitions=args.partitions,
        )
        observed = Counter(str(row["partition"]) for row in rows)
        if dict(observed) != {
            partition: validated_split["case_counts"][partition]
            for partition in sorted(PARTITION_SET)
        }:
            parser.error("materialized partition counts differ from validated learning split")
        input_sha256 = {
            "identity_manifest": sha256_file(identity_path),
            "learning_split": validated_split["split_sha256"],
            "experiment_config": sha256_file(experiment_config_path),
            "materialized_inventory": inventory_sha,
        }
        if access_receipt is not None:
            input_sha256["test_access_receipt"] = sha256_file(args.test_access_receipt)

    manifest_sha = publish_jsonl(rows, args.output)
    print(
        json.dumps(
            {
                "schema_version": RECEIPT_SCHEMA,
                "status": "PASS",
                "mode": args.mode,
                "authorized_partitions": (
                    [] if args.mode == IDENTITY_MODE else sorted(set(args.partitions))
                ),
                "case_count": len(rows),
                "patient_count": len({row["patient_id"] for row in rows}),
                "fold_count": EXPECTED_FOLDS,
                "materialized_case_count": sum(
                    row["truth_materialization"] == MATERIALIZED_STATE for row in rows
                ),
                "locked_unread_case_count": sum(
                    row["truth_materialization"] == LOCKED_STATE for row in rows
                ),
                "input_sha256": input_sha256,
                "output": str(args.output.resolve()),
                "manifest_sha256": manifest_sha,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
