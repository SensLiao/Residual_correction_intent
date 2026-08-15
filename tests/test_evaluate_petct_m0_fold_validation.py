from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "scripts"))

from evaluation.evaluate_petct_m0_fold_validation import evaluate_folds, publish_evaluation  # noqa: E402


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, allow_nan=True, indent=2) + "\n", encoding="utf-8")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _record(path: Path, *, relative: bool = False, root: Path | None = None) -> dict[str, object]:
    return {
        "path": str(path.relative_to(root)) if relative and root is not None else str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": _sha(path),
    }


def _case(patient: int, visit: int) -> str:
    return f"psma_{patient:016x}_2020-01-{visit + 1:02d}"


def _fixture(tmp_path: Path) -> Path:
    campaign = tmp_path / "petct_m0_5fold_fixture"
    campaign.mkdir(parents=True)
    cases = [_case(patient, visit) for patient in range(5) for visit in range(2)]
    splits = []
    for fold in range(5):
        val = [case_id for case_id in cases if case_id.startswith(f"psma_{fold:016x}_")]
        splits.append({"train": [case_id for case_id in cases if case_id not in val], "val": val})
    split_path = tmp_path / "splits_final.json"
    _write_json(split_path, splits)
    spec = {
        "status": "STAGED",
        "phase": "STANDARD_5FOLD_FULL_TRAINING",
        "campaign_id": campaign.name,
        "full_training_status": "NOT_STARTED",
        "oof_status": "NOT_STARTED",
        "oof_prediction_count": 0,
        "result_count": 0,
        "thesis_citable": False,
        "prerequisite_paths": {"splits_final": str(split_path.resolve())},
        "prerequisite_bound_hashes": {"splits_final": _sha(split_path)},
    }
    spec_path = campaign / "CAMPAIGN_SPEC.json"
    _write_json(spec_path, spec)
    for fold in (0, 1):
        artifacts = {"validation_masks": [], "validation_probabilities": [], "validation_properties": []}
        metric_rows = []
        for index, case_id in enumerate(splits[fold]["val"]):
            validation = campaign / "validation" / f"fold_{fold}"
            prediction = validation / f"{case_id}.nii.gz"
            probability = validation / f"{case_id}.npz"
            properties = validation / f"{case_id}.pkl"
            reference = tmp_path / "gt" / f"{case_id}.nii.gz"
            for path, content in ((prediction, b"pred"), (probability, b"prob"), (properties, b"props"), (reference, b"gt")):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content)
            artifacts["validation_masks"].append(_record(prediction, relative=True, root=campaign))
            artifacts["validation_probabilities"].append(_record(probability, relative=True, root=campaign))
            artifacts["validation_properties"].append(_record(properties, relative=True, root=campaign))
            tp, fp, fn, tn = 8 + index, 1, 2, 20
            n_ref, n_pred = tp + fn, tp + fp
            metric_rows.append({
                "prediction_file": str(prediction.resolve()),
                "reference_file": str(reference.resolve()),
                "metrics": {"1": {
                    "Dice": 2 * tp / (n_ref + n_pred),
                    "IoU": tp / (tp + fp + fn),
                    "TP": tp, "FP": fp, "FN": fn, "TN": tn,
                    "n_pred": n_pred, "n_ref": n_ref,
                }},
            })
        dice = [row["metrics"]["1"]["Dice"] for row in metric_rows]
        summary_path = campaign / "validation" / f"fold_{fold}" / "summary.json"
        _write_json(summary_path, {"metric_per_case": metric_rows, "foreground_mean": {"Dice": sum(dice) / len(dice)}})
        artifacts["validation_summary"] = _record(summary_path, relative=True, root=campaign)
        output_contract = {
            "status": "PASS", "fold": fold, "actual_validation": True,
            "export_probabilities": True, "oof_handoff_inputs_present": True,
            "oof_publication_count": 0, "result_publication_count": 0,
            "artifacts": artifacts,
        }
        receipt = {
            "status": "COMMITTED", "phase": "STANDARD_5FOLD_FULL_TRAINING",
            "campaign_id": campaign.name, "fold": fold,
            "full_fold_training_status": "PASS", "oof_status": "NOT_STARTED",
            "oof_prediction_count": 0, "result_count": 0, "thesis_citable": False,
            "campaign_spec": _record(spec_path),
            "prerequisite_bound_hashes": spec["prerequisite_bound_hashes"],
            "output_contract": output_contract,
        }
        _write_json(campaign / "fold_receipts" / f"fold_{fold}.json", receipt)
    return campaign


def test_fold_validation_qa_binds_receipts_splits_and_metric_identities(tmp_path: Path) -> None:
    campaign = _fixture(tmp_path)
    records, completion = evaluate_folds(campaign, (0, 1))
    assert [record["fold"] for record in records] == [0, 1]
    assert all(record["patient_exclusion"] == "PASS" for record in records)
    assert all(record["scientific_role"] == "FOLD_LOCAL_VALIDATION_DIAGNOSTIC_ONLY" for record in records)
    assert completion["case_count"] == 4
    assert completion["thesis_citable"] is False

    output = tmp_path / "evaluation" / "fold01"
    receipt = publish_evaluation(output, records, completion)
    published = json.loads(receipt.read_text(encoding="utf-8"))
    assert published["status"] == "COMPLETE"
    assert published["folds_evaluated"] == [0, 1]
    assert len(published["fold_receipts"]) == 2
    assert not list(output.parent.glob(".fold01.partial-*"))


def test_fold_validation_qa_fails_closed_on_metric_or_split_drift(tmp_path: Path) -> None:
    campaign = _fixture(tmp_path)
    summary_path = campaign / "validation" / "fold_0" / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["metric_per_case"][0]["metrics"]["1"]["Dice"] = 0.99
    _write_json(summary_path, summary)
    with pytest.raises(RuntimeError, match="changed after receipt"):
        evaluate_folds(campaign, (0,))

    campaign = _fixture(tmp_path / "second")
    split_path = Path(json.loads((campaign / "CAMPAIGN_SPEC.json").read_text(encoding="utf-8"))["prerequisite_paths"]["splits_final"])
    split_path.write_text(split_path.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(RuntimeError, match="splits_final changed"):
        evaluate_folds(campaign, (0,))


def test_fold_validation_qa_refuses_overwrite(tmp_path: Path) -> None:
    campaign = _fixture(tmp_path)
    records, completion = evaluate_folds(campaign, (0,))
    output = tmp_path / "evaluation" / "fold0"
    publish_evaluation(output, records, completion)
    with pytest.raises(FileExistsError, match="overwrite"):
        publish_evaluation(output, records, completion)
