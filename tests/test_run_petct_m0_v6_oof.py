from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
SCRIPTS = PROJECT / "scripts"
for directory in (SCRIPTS, SCRIPTS / "baseline"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import run_petct_m0_v6_oof as oof  # noqa: E402
from common.petct_mainline_lineage import (  # noqa: E402
    MAINLINE_SOURCE,
    file_record,
    validate_m0_v6_oof_ready,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _fixture(tmp_path: Path):
    shared = tmp_path / "image.nii.gz"
    shared.write_bytes(b"bound-input")
    cases = []
    patient_cases = []
    case_index = 0
    for patient_index in range(321):
        count = 2 if patient_index < 185 else 1
        fold = patient_index % 5
        members = []
        for _ in range(count):
            case_id = f"case-{case_index:03d}"
            case_index += 1
            members.append(case_id)
            cases.append(
                {
                    "case_id": case_id,
                    "patient_id": f"patient-{patient_index:03d}",
                    # Historical source manifests may carry the pre-v6 fold.
                    # The explicit v6 splits_final is the only fold authority.
                    "held_out_fold": (fold + 1) % 5,
                    "ct_path": str(shared),
                    "ct_sha256": _sha(shared),
                    "ct_bytes": shared.stat().st_size,
                    "pet_path": str(shared),
                    "pet_sha256": _sha(shared),
                    "pet_bytes": shared.stat().st_size,
                    "gt_path": str(shared),
                    "gt_sha256": _sha(shared),
                    "gt_bytes": shared.stat().st_size,
                }
            )
        patient_cases.append((fold, members))
    assert len(cases) == 506
    universe = [row["case_id"] for row in cases]
    splits = []
    for fold in range(5):
        val = [case for assigned, members in patient_cases if assigned == fold for case in members]
        splits.append({"train": [case for case in universe if case not in set(val)], "val": val})
    split_path = tmp_path / "splits_final.json"
    _write_json(split_path, {str(index): value for index, value in enumerate(splits)})
    source_manifest = tmp_path / "source_cases.jsonl"
    source_manifest.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in cases),
        encoding="utf-8",
    )
    model_root = tmp_path / "model"
    model_root.mkdir()
    (model_root / "plans.json").write_text("{}", encoding="utf-8")
    (model_root / "dataset.json").write_text("{}", encoding="utf-8")
    for fold in range(5):
        folder = model_root / f"fold_{fold}"
        folder.mkdir()
        (folder / "checkpoint_final.pth").write_bytes(f"fold-{fold}".encode())
    return split_path, source_manifest, model_root


def test_m0_v6_oof_stage_and_finalize_publish_only_506_learning_cases(
    tmp_path: Path,
) -> None:
    split_path, source_manifest, model_root = _fixture(tmp_path)
    run_root = tmp_path / "PETCT-M0-V6-OOF-R1"
    ready = tmp_path / "M0_V6_FIVEFOLD_OOF_READY.json"
    plan = oof.stage(
        run_root=run_root,
        ready=ready,
        splits_final=split_path,
        source_manifest=source_manifest,
        model_root=model_root,
    )
    assert plan["source_m0_lineage"] == MAINLINE_SOURCE
    assert plan["case_count"] == 506
    assert plan["locked_test_present"] is False

    for fold in range(5):
        rows = []
        for source in (row for row in plan["cases"] if row["held_out_fold"] == fold):
            mask = run_root / "outputs" / f"fold_{fold}" / "masks" / f"{source['case_id']}.nii.gz"
            probability = run_root / "outputs" / f"fold_{fold}" / "probabilities" / f"{source['case_id']}.npz"
            mask.write_bytes(b"mask")
            probability.write_bytes(b"probability")
            rows.append(
                {
                    "case_id": source["case_id"],
                    "mask": file_record(mask),
                    "foreground_probability": file_record(probability),
                }
            )
        _write_json(
            run_root / "outputs" / f"fold_{fold}" / "FOLD_DONE.json",
            {
                "schema_version": oof.FOLD_SCHEMA,
                "status": "PASS",
                "source_m0_lineage": MAINLINE_SOURCE,
                "fold": fold,
                "prediction_count": len(rows),
                "cases": rows,
            },
        )
    oof.finalize(run_root=run_root, ready=ready)
    validated = validate_m0_v6_oof_ready(ready)
    assert len(validated["cases"]) == 506
    assert validated["locked_test_present"] is False
