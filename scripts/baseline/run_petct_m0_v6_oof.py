#!/usr/bin/env python3
"""Build the clean 506-case M0-v6 patient-excluded OOF receipt.

This is intentionally separate from the historical 597-case OOF campaign.
Only the 506 learning cases are predicted; the 91 locked TEST cases are absent
from plans, outputs, and receipts.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import nibabel as nib

SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from common.petct_mainline_lineage import (  # noqa: E402
    MAINLINE_SOURCE,
    M0_V6_OOF_SCHEMA,
    LineageContractError,
    file_record,
    validate_m0_v6_oof_ready,
)
from baseline.run_petct_m0_oof_fold import (  # noqa: E402
    OFFICIAL_METADATA,
    _extract_foreground_probability,
    _official_predictor_factory,
)


BUNDLE_SCHEMA = "PETCT-M0-V6-OOF-BUNDLE-v1.0"
PLAN_SCHEMA = "PETCT-M0-V6-OOF-PLAN-v1.0"
FOLD_SCHEMA = "PETCT-M0-V6-OOF-FOLD-v1.0"


def _load_json(path: Path, label: str) -> Any:
    if path.is_symlink() or not path.is_file():
        raise LineageContractError(f"{label} must be a regular file")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LineageContractError(f"{label} is invalid JSON") from exc


def _load_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if path.is_symlink() or not path.is_file():
        raise LineageContractError(f"{label} must be a regular file")
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise LineageContractError(
                    f"{label} line {line_number} is invalid"
                ) from exc
            if not isinstance(row, dict):
                raise LineageContractError(f"{label} line {line_number} is not an object")
            rows.append(row)
    return rows


def _write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _validate_splits(document: Any) -> tuple[list[dict[str, list[str]]], dict[str, int]]:
    if isinstance(document, dict):
        if set(document) != {"0", "1", "2", "3", "4"}:
            raise LineageContractError("M0 v6 split mapping must have keys 0..4")
        document = [document[str(fold)] for fold in range(5)]
    if not isinstance(document, list) or len(document) != 5:
        raise LineageContractError("M0 v6 splits_final must contain five folds")
    normalized: list[dict[str, list[str]]] = []
    case_to_fold: dict[str, int] = {}
    universe: set[str] | None = None
    for fold, item in enumerate(document):
        if not isinstance(item, dict) or set(item) != {"train", "val"}:
            raise LineageContractError(f"fold {fold} must contain train/val only")
        train = [str(value) for value in item["train"]]
        val = [str(value) for value in item["val"]]
        if (
            len(train) != len(set(train))
            or len(val) != len(set(val))
            or set(train) & set(val)
        ):
            raise LineageContractError(f"fold {fold} has duplicate or overlapping cases")
        current = set(train) | set(val)
        if universe is None:
            universe = current
        elif current != universe:
            raise LineageContractError("M0 v6 fold universes differ")
        for case_id in val:
            if case_id in case_to_fold:
                raise LineageContractError("M0 v6 case is held out more than once")
            case_to_fold[case_id] = fold
        normalized.append({"train": train, "val": val})
    if universe is None or len(universe) != 506 or set(case_to_fold) != universe:
        raise LineageContractError("M0 v6 split must cover 506 cases exactly once")
    return normalized, case_to_fold


def stage(
    *,
    run_root: Path,
    ready: Path,
    splits_final: Path,
    source_manifest: Path,
    model_root: Path,
) -> dict[str, Any]:
    run_root, ready = run_root.resolve(), ready.resolve()
    if run_root.exists() or run_root.is_symlink():
        raise LineageContractError("refusing existing M0 v6 OOF run root")
    if ready.exists() or ready.is_symlink():
        raise LineageContractError("refusing existing M0 v6 OOF receipt")
    splits, case_to_fold = _validate_splits(_load_json(splits_final, "splits_final"))
    source_rows = _load_jsonl(source_manifest, "source case manifest")
    source_by_case = {str(row.get("case_id") or ""): row for row in source_rows}
    if "" in source_by_case or len(source_by_case) != len(source_rows):
        raise LineageContractError("source case manifest has missing/duplicate case IDs")
    missing = sorted(set(case_to_fold) - set(source_by_case))
    if missing:
        raise LineageContractError(f"source manifest misses M0 v6 cases: {missing[:3]}")
    patients: defaultdict[str, set[int]] = defaultdict(set)
    cases: list[dict[str, Any]] = []
    for case_id, fold in sorted(case_to_fold.items()):
        row = source_by_case[case_id]
        patient_id = str(row.get("patient_id") or "").casefold()
        if not patient_id:
            raise LineageContractError("source patient identity is missing")
        patients[patient_id].add(fold)
        for modality in ("ct", "pet", "gt"):
            path = Path(str(row.get(f"{modality}_path") or ""))
            if path.is_symlink() or not path.is_file():
                raise LineageContractError(f"missing {modality} input for {case_id}")
            if not row.get(f"{modality}_sha256") or not row.get(f"{modality}_bytes"):
                raise LineageContractError(f"unbound {modality} input for {case_id}")
        cases.append(
            {
                "case_id": case_id,
                "patient_id": patient_id,
                "held_out_fold": fold,
                "source_manifest_held_out_fold": int(row.get("held_out_fold", -1)),
                "ct_path": str(Path(row["ct_path"]).resolve()),
                "ct_sha256": str(row["ct_sha256"]),
                "ct_bytes": int(row["ct_bytes"]),
                "pet_path": str(Path(row["pet_path"]).resolve()),
                "pet_sha256": str(row["pet_sha256"]),
                "pet_bytes": int(row["pet_bytes"]),
                "gt_path": str(Path(row["gt_path"]).resolve()),
                "gt_sha256": str(row["gt_sha256"]),
                "gt_bytes": int(row["gt_bytes"]),
            }
        )
    if len(patients) != 321 or any(len(value) != 1 for value in patients.values()):
        raise LineageContractError("M0 v6 learning patients are not fold-disjoint")
    model_root = model_root.resolve()
    checkpoints = []
    for fold in range(5):
        checkpoint = model_root / f"fold_{fold}" / "checkpoint_final.pth"
        checkpoints.append({"fold": fold, "checkpoint": file_record(checkpoint)})
    for metadata in (model_root / "plans.json", model_root / "dataset.json"):
        if metadata.is_symlink() or not metadata.is_file():
            raise LineageContractError(f"missing nnU-Net metadata: {metadata}")
    run_root.mkdir(parents=True)
    for fold in range(5):
        (run_root / "outputs" / f"fold_{fold}" / "masks").mkdir(parents=True)
        (run_root / "outputs" / f"fold_{fold}" / "probabilities").mkdir(parents=True)
    plan = {
        "schema_version": PLAN_SCHEMA,
        "status": "STAGED",
        "source_m0_lineage": MAINLINE_SOURCE,
        "mainline_eligible": True,
        "lifecycle": "active",
        "patient_excluded": True,
        "locked_test_present": False,
        "case_count": 506,
        "patient_count": 321,
        "fold_count": 5,
        "checkpoint_selector": "checkpoint_final.pth",
        "fold_authority": "explicit M0-v6 splits_final; source-manifest fold annotation ignored",
        "splits_final": file_record(splits_final),
        "source_manifest": file_record(source_manifest),
        "model_root": str(model_root),
        "model_metadata": {
            "plans": file_record(model_root / "plans.json"),
            "dataset_json": file_record(model_root / "dataset.json"),
        },
        "checkpoints": checkpoints,
        "splits": splits,
        "cases": cases,
        "ready_path": str(ready),
    }
    _write_json_exclusive(run_root / "OOF_PLAN.json", plan)
    return plan


def run_fold(*, run_root: Path, fold: int, device: str) -> dict[str, Any]:
    run_root = run_root.resolve()
    plan = _load_json(run_root / "OOF_PLAN.json", "M0 v6 OOF plan")
    if plan.get("schema_version") != PLAN_SCHEMA or plan.get("status") != "STAGED":
        raise LineageContractError("M0 v6 OOF plan is not staged")
    done = run_root / "outputs" / f"fold_{fold}" / "FOLD_DONE.json"
    if done.exists() or done.is_symlink():
        raise LineageContractError(f"fold {fold} already has a completion receipt")
    mask_root = done.parent / "masks"
    probability_root = done.parent / "probabilities"
    if any(mask_root.iterdir()) or any(probability_root.iterdir()):
        raise LineageContractError(f"fold {fold} output root is not fresh")
    cases = [row for row in plan["cases"] if int(row["held_out_fold"]) == fold]
    expected = set(plan["splits"][fold]["val"])
    if {row["case_id"] for row in cases} != expected:
        raise LineageContractError(f"fold {fold} plan differs from val split")
    inputs = [[row["ct_path"], row["pet_path"]] for row in cases]
    outputs = [str(mask_root / row["case_id"]) for row in cases]
    predictor = _official_predictor_factory(device)
    predictor.initialize_from_trained_model_folder(
        str(plan["model_root"]),
        use_folds=(fold,),
        checkpoint_name="checkpoint_final.pth",
    )
    predictor.predict_from_files(
        inputs,
        outputs,
        save_probabilities=True,
        overwrite=False,
        num_processes_preprocessing=2,
        num_processes_segmentation_export=2,
    )
    records = []
    for row in cases:
        case_id = row["case_id"]
        mask = mask_root / f"{case_id}.nii.gz"
        raw_probability = mask_root / f"{case_id}.npz"
        properties = mask_root / f"{case_id}.pkl"
        if not mask.is_file() or mask.is_symlink():
            raise LineageContractError(f"fold {fold} missed mask {case_id}")
        shape = tuple(int(value) for value in nib.load(str(mask)).shape)
        probability = probability_root / f"{case_id}.npz"
        _extract_foreground_probability(
            raw_probability, probability, expected_mask_shape=shape
        )
        raw_probability.unlink()
        if properties.is_file() and not properties.is_symlink():
            properties.unlink()
        records.append(
            {
                "case_id": case_id,
                "mask": file_record(mask),
                "foreground_probability": file_record(probability),
            }
        )
    for name in OFFICIAL_METADATA:
        metadata = mask_root / name
        if metadata.is_file() and not metadata.is_symlink():
            metadata.unlink()
    if {path.stem.removesuffix(".nii") for path in mask_root.glob("*.nii.gz")} != expected:
        raise LineageContractError(f"fold {fold} mask inventory differs from plan")
    payload = {
        "schema_version": FOLD_SCHEMA,
        "status": "PASS",
        "source_m0_lineage": MAINLINE_SOURCE,
        "fold": fold,
        "prediction_count": len(records),
        "cases": records,
    }
    _write_json_exclusive(done, payload)
    return payload


def finalize(*, run_root: Path, ready: Path) -> dict[str, Any]:
    run_root, ready = run_root.resolve(), ready.resolve()
    if ready.exists() or ready.is_symlink():
        raise LineageContractError("refusing existing M0 v6 OOF receipt")
    plan = _load_json(run_root / "OOF_PLAN.json", "M0 v6 OOF plan")
    plan_cases = {row["case_id"]: row for row in plan["cases"]}
    outputs: dict[str, dict[str, Any]] = {}
    for fold in range(5):
        done_path = run_root / "outputs" / f"fold_{fold}" / "FOLD_DONE.json"
        done = _load_json(done_path, f"fold {fold} receipt")
        if (
            done.get("schema_version") != FOLD_SCHEMA
            or done.get("status") != "PASS"
            or done.get("fold") != fold
        ):
            raise LineageContractError(f"fold {fold} receipt is invalid")
        for row in done["cases"]:
            case_id = str(row["case_id"])
            if case_id in outputs or case_id not in plan_cases:
                raise LineageContractError("M0 v6 OOF output coverage is invalid")
            outputs[case_id] = row
    if set(outputs) != set(plan_cases) or len(outputs) != 506:
        raise LineageContractError("M0 v6 OOF does not cover 506 cases exactly once")
    cases = []
    for case_id in sorted(plan_cases):
        source, output = plan_cases[case_id], outputs[case_id]
        mask = Path(output["mask"]["path"]).resolve()
        probability = Path(output["foreground_probability"]["path"]).resolve()
        cases.append(
            {
                "case_id": case_id,
                "patient_id": source["patient_id"],
                "held_out_fold": source["held_out_fold"],
                "mask": {
                    **output["mask"],
                    "path": mask.relative_to(run_root).as_posix(),
                },
                "foreground_probability": {
                    **output["foreground_probability"],
                    "path": probability.relative_to(run_root).as_posix(),
                },
                "input_ct_sha256": source["ct_sha256"],
                "input_ct_bytes": source["ct_bytes"],
                "input_pet_sha256": source["pet_sha256"],
                "input_pet_bytes": source["pet_bytes"],
                "input_gt_sha256": source["gt_sha256"],
                "input_gt_bytes": source["gt_bytes"],
            }
        )
    bundle = {
        "schema_version": BUNDLE_SCHEMA,
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
        "splits_final": plan["splits_final"],
        "source_manifest": plan["source_manifest"],
        "checkpoints": plan["checkpoints"],
        "cases": cases,
    }
    bundle_path = run_root / "OOF_BUNDLE.json"
    _write_json_exclusive(bundle_path, bundle)
    receipt = {
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
        "run_dir": str(run_root),
        "run_receipt": file_record(bundle_path),
    }
    _write_json_exclusive(ready, receipt)
    validate_m0_v6_oof_ready(ready)
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    stage_parser = commands.add_parser("stage")
    stage_parser.add_argument("--run-root", type=Path, required=True)
    stage_parser.add_argument("--ready", type=Path, required=True)
    stage_parser.add_argument("--splits-final", type=Path, required=True)
    stage_parser.add_argument("--source-manifest", type=Path, required=True)
    stage_parser.add_argument("--model-root", type=Path, required=True)
    fold_parser = commands.add_parser("run-fold")
    fold_parser.add_argument("--run-root", type=Path, required=True)
    fold_parser.add_argument("--fold", type=int, choices=range(5), required=True)
    fold_parser.add_argument("--device", default="cuda:0")
    finalize_parser = commands.add_parser("finalize")
    finalize_parser.add_argument("--run-root", type=Path, required=True)
    finalize_parser.add_argument("--ready", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "stage":
        payload = stage(
            run_root=args.run_root,
            ready=args.ready,
            splits_final=args.splits_final,
            source_manifest=args.source_manifest,
            model_root=args.model_root,
        )
    elif args.command == "run-fold":
        payload = run_fold(run_root=args.run_root, fold=args.fold, device=args.device)
    else:
        payload = finalize(run_root=args.run_root, ready=args.ready)
    print(json.dumps({"status": payload["status"], "source_m0_lineage": MAINLINE_SOURCE}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
