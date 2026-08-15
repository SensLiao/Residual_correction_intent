from __future__ import annotations

import copy
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(SCRIPTS / "data"))

from build_petct_development_manifest import (  # noqa: E402
    DATASET_ID,
    PATIENT_SPLIT_AUTHORITY_VERSION,
    RULE_VERSION,
    STRATEGY_SALT,
    DevelopmentManifestError,
    build_development_manifests,
    compile_development_manifests,
    validate_development_manifest_documents,
    validate_published_development_manifests,
)
from build_petct_scribble_episode import assign_scribble_strategy  # noqa: E402


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _asset(patient_id: str, case_id: str, modality: str) -> dict[str, str]:
    return {
        "path": f"/srv/psma/{patient_id}/{case_id}_{modality}.nii.gz",
        "sha256": _digest(f"{patient_id}|{case_id}|{modality}"),
    }


def _case(
    patient_number: int,
    exam_number: int = 1,
    *,
    split: str = "development",
    tracer: str = "PSMA",
) -> dict[str, Any]:
    patient_id = f"secret-patient-{patient_number:02d}"
    case_id = f"secret-case-{patient_number:02d}-{exam_number:02d}"
    return {
        "case_id": case_id,
        "patient_id": patient_id,
        "status": "PASS",
        "dataset_id": DATASET_ID,
        "dataset_scope": "PSMA",
        "tracer": tracer,
        "split": split,
        "source_assets": {
            modality: _asset(patient_id, case_id, modality)
            for modality in ("pet", "ct", "gt")
        },
    }


def _json_digest(document: dict[str, Any]) -> str:
    return hashlib.sha256(
        (json.dumps(document, indent=2, sort_keys=True) + "\n").encode("utf-8")
    ).hexdigest()


def _source_bundle(
    patient_count: int = 12,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    records = [_case(index) for index in range(patient_count)]
    # Repeated examinations are legal and must remain clustered by patient.
    records.extend([_case(2, 2), _case(2, 3), _case(7, 2)])
    audit_report_sha256 = _digest("authoritative-nifti-audit-report")
    audit_receipt = {
        "status": "COMMITTED",
        "audit_status": "PASS",
        "audit_version": "1.2.0",
        "outputs": {
            "psma_v3_nifti_audit.json": {
                "bytes": 123456,
                "sha256": audit_report_sha256,
            }
        },
    }
    audit_receipt_sha256 = _json_digest(audit_receipt)
    patients: dict[str, list[str]] = {}
    for record in records:
        patients.setdefault(record["patient_id"], []).append(record["case_id"])
    split_manifest = {
        "schema_version": PATIENT_SPLIT_AUTHORITY_VERSION,
        "status": "FROZEN_CONTRACT_ONLY",
        "dataset_id": DATASET_ID,
        "dataset_scope": "PSMA",
        "split_unit": "patient",
        "patient_disjoint": True,
        "source_audit_receipt_sha256": audit_receipt_sha256,
        "source_nifti_audit_sha256": audit_report_sha256,
        "full_cohort_patient_count": len(patients),
        "full_cohort_case_count": len(records),
        "patients": [
            {
                "patient_id": patient_id,
                "partition": "development_pool",
                "case_ids": sorted(case_ids),
            }
            for patient_id, case_ids in sorted(patients.items())
        ],
    }
    source = {
        "schema_version": "PETCT-PSMA-CASE-AUDIT-EXPORT-v1.0",
        "audit_status": "PASS",
        "dataset_id": DATASET_ID,
        "dataset_scope": "PSMA",
        "split_name": "development_pool",
        "split_unit": "patient",
        "patient_disjoint": True,
        "source_audit_receipt_sha256": audit_receipt_sha256,
        "source_nifti_audit_sha256": audit_report_sha256,
        "source_split_manifest_sha256": _json_digest(split_manifest),
        "case_count": len(records),
        "patient_count": patient_count,
        "case_records": records,
    }
    for record in source["case_records"]:
        record["split"] = "development_pool"
    return source, audit_receipt, split_manifest


def _write_inputs(
    tmp_path: Path,
    bundle: tuple[dict[str, Any], dict[str, Any], dict[str, Any]],
) -> tuple[Path, Path, Path]:
    paths = (
        tmp_path / "case-audit-export.json",
        tmp_path / "AUDIT_COMPLETE.json",
        tmp_path / "patient-split-authority.json",
    )
    for path, document in zip(paths, bundle):
        path.write_bytes(
            (json.dumps(document, indent=2, sort_keys=True) + "\n").encode("utf-8")
        )
    return paths


def _compile(
    bundle: tuple[dict[str, Any], dict[str, Any], dict[str, Any]],
    input_digest: str = _digest("input"),
) -> dict[str, Any]:
    source, audit_receipt, split_manifest = bundle
    return compile_development_manifests(
        source,
        input_digest,
        audit_receipt_document=audit_receipt,
        audit_receipt_sha256=_json_digest(audit_receipt),
        split_document=split_manifest,
        split_input_sha256=_json_digest(split_manifest),
    )


def _walk_keys(value: Any) -> list[str]:
    keys: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            keys.append(str(key).casefold())
            keys.extend(_walk_keys(child))
    elif isinstance(value, list):
        for child in value:
            keys.extend(_walk_keys(child))
    return keys


def test_builds_deterministic_patient_manifest_with_4_3_3_scribble_strata(
    tmp_path: Path,
) -> None:
    bundle = _source_bundle()
    source, _, _ = bundle
    source_path, audit_receipt_path, split_path = _write_inputs(tmp_path, bundle)

    receipt = build_development_manifests(
        source_path,
        audit_receipt_path=audit_receipt_path,
        patient_split_path=split_path,
        visible_root=tmp_path / "visible",
        eval_root=tmp_path / "eval",
    )

    public_path = Path(receipt["public_manifest_path"])
    private_path = Path(receipt["private_manifest_path"])
    public = json.loads(public_path.read_text(encoding="utf-8"))
    private = json.loads(private_path.read_text(encoding="utf-8"))
    public_text = json.dumps(public, sort_keys=True)

    assert receipt["status"] == "SELECTED_NOT_MATERIALIZED"
    assert receipt["selected_patient_count"] == 10
    assert public_path.parent == (tmp_path / "visible").resolve()
    assert private_path.parent == (tmp_path / "eval").resolve()
    assert public["selection_rule_version"] == RULE_VERSION
    assert public["selected_patient_count"] == 10
    assert Counter(row["scribble_strategy"] for row in public["patients"]) == {
        "centerline": 4,
        "random": 3,
        "boundary": 3,
    }
    assert public["prompt_contract"] == {
        "modality": "scribble",
        "polarity": "foreground",
        "strategies": ["centerline", "random", "boundary"],
    }
    assert all(
        row["public_patient_id"].startswith("DEV-P") for row in public["patients"]
    )
    assert len({row["public_patient_id"] for row in public["patients"]}) == 10
    assert len({row["patient_id"] for row in private["patients"]}) == 10
    assert not any(
        record["patient_id"] in public_text or record["case_id"] in public_text
        for record in source["case_records"]
    )
    assert "/srv/psma" not in public_text
    assert "source_assets" not in public_text
    assert "case_id" not in public_text
    assert not any(
        token in key.replace("-", "_").split("_")
        for key in _walk_keys(public)
        for token in ("click", "clicks", "point", "points")
    )
    repeated = next(
        row for row in private["patients"] if row["patient_id"] == "secret-patient-02"
    )
    assert repeated["case_count"] == 3
    assert len(repeated["cases"]) == 3

    validation = validate_published_development_manifests(
        source_path=source_path,
        audit_receipt_path=audit_receipt_path,
        patient_split_path=split_path,
        public_manifest_path=public_path,
        private_manifest_path=private_path,
        receipt_path=Path(receipt["receipt_path"]),
    )
    assert validation["status"] == "PASS"


def test_selection_and_strategy_are_order_invariant() -> None:
    bundle = _source_bundle()
    reversed_bundle = copy.deepcopy(bundle)
    reversed_source = reversed_bundle[0]
    reversed_source["case_records"].reverse()
    input_digest = _digest("same-byte-provenance-for-unit-test")

    first = _compile(bundle, input_digest)
    second = _compile(reversed_bundle, input_digest)

    def assignments(artifacts: dict[str, Any]) -> dict[str, str]:
        return {
            row["patient_id"]: row["scribble_strategy"]
            for row in artifacts["private"]["patients"]
        }

    assert assignments(first) == assignments(second)
    assert first == second


def test_selection_rule_is_frozen_sha256_within_shared_strategy_strata() -> None:
    bundle = _source_bundle(patient_count=15)
    source = bundle[0]
    artifacts = _compile(bundle)
    expected = []
    for strategy, quota in (("centerline", 4), ("random", 3), ("boundary", 3)):
        stratum = {
            patient["patient_id"]
            for patient in source["case_records"]
            if assign_scribble_strategy(patient["patient_id"], salt=STRATEGY_SALT)
            == strategy
        }
        expected.extend(
            sorted(
                stratum,
                key=lambda patient_id: (
                    _digest(f"{RULE_VERSION}|select|{patient_id}"),
                    patient_id,
                ),
            )[:quota]
        )
    expected.sort(
        key=lambda patient_id: (
            _digest(f"{RULE_VERSION}|select|{patient_id}"),
            patient_id,
        )
    )

    assert [row["patient_id"] for row in artifacts["private"]["patients"]] == expected


def test_strategy_authority_and_repeated_exam_priority_are_downstream_compatible() -> (
    None
):
    artifacts = _compile(_source_bundle())

    for patient in artifacts["private"]["patients"]:
        assert patient["scribble_strategy"] == assign_scribble_strategy(
            patient["patient_id"],
            salt=STRATEGY_SALT,
        )
        priorities = [case["case_priority"] for case in patient["cases"]]
        assert priorities == list(range(1, len(priorities) + 1))
        assert patient["primary_case_id"] == patient["cases"][0]["case_id"]
        assert [case["case_priority_sha256"] for case in patient["cases"]] == sorted(
            case["case_priority_sha256"] for case in patient["cases"]
        )


def test_binds_authoritative_audit_receipt_and_full_patient_split() -> None:
    bundle = _source_bundle()
    tampered_receipt = copy.deepcopy(bundle)
    tampered_receipt[1]["outputs"]["psma_v3_nifti_audit.json"]["sha256"] = _digest(
        "tampered-audit"
    )
    incomplete_split = copy.deepcopy(bundle)
    removed = incomplete_split[2]["patients"].pop()
    incomplete_split[2]["full_cohort_patient_count"] -= 1
    incomplete_split[2]["full_cohort_case_count"] -= len(removed["case_ids"])
    incomplete_split[0]["source_split_manifest_sha256"] = _json_digest(
        incomplete_split[2]
    )

    with pytest.raises(DevelopmentManifestError, match="audit receipt hash mismatch"):
        _compile(tampered_receipt)
    with pytest.raises(DevelopmentManifestError, match="development pool patient set"):
        _compile(incomplete_split)


@pytest.mark.parametrize(
    "mutation,error",
    [
        (lambda doc: doc.update(dataset_scope="FDG"), "PSMA-only"),
        (lambda doc: doc.update(dataset_id="autoPET-FDG"), "PSMA-only"),
        (lambda doc: doc["case_records"][0].update(tracer="FDG"), "FDG"),
        (lambda doc: doc.update(split_name="test"), "development-only"),
        (lambda doc: doc.update(split_unit="case"), "patient-level"),
        (lambda doc: doc.update(patient_disjoint=False), "patient-disjoint"),
        (lambda doc: doc["case_records"][0].update(split="locked"), "development-only"),
    ],
)
def test_rejects_non_psma_non_development_or_case_level_split(
    mutation: Any,
    error: str,
) -> None:
    bundle = _source_bundle()
    source = bundle[0]
    mutation(source)

    with pytest.raises(DevelopmentManifestError, match=error):
        _compile(bundle)


@pytest.mark.parametrize(
    "field,value",
    [
        ("prompt_modality", "click"),
        ("prompt_type", "point"),
        ("point_3d", [1, 2, 3]),
        ("click_prompt", {"count": 1}),
    ],
)
def test_rejects_click_or_point_prompt_semantics(field: str, value: Any) -> None:
    bundle = _source_bundle()
    source = bundle[0]
    source["case_records"][0][field] = value

    with pytest.raises(DevelopmentManifestError, match="scribble-only"):
        _compile(bundle)


@pytest.mark.parametrize(
    "mutation,error",
    [
        (lambda doc: doc["case_records"][0].pop("patient_id"), "patient_id"),
        (lambda doc: doc.pop("source_audit_receipt_sha256"), "source audit receipt"),
        (
            lambda doc: doc["case_records"][0]["source_assets"]["pet"].pop("sha256"),
            "source asset hash",
        ),
        (
            lambda doc: doc["case_records"][0]["source_assets"].pop("gt"),
            "source asset",
        ),
        (lambda doc: doc.update(audit_status="FAIL"), "audit_status"),
    ],
)
def test_rejects_missing_patient_or_source_provenance(
    mutation: Any,
    error: str,
) -> None:
    bundle = _source_bundle()
    source = bundle[0]
    mutation(source)

    with pytest.raises(DevelopmentManifestError, match=error):
        _compile(bundle)


def test_rejects_duplicate_case_but_clusters_repeated_patient() -> None:
    bundle = _source_bundle()
    source = bundle[0]
    duplicate = copy.deepcopy(source["case_records"][0])
    duplicate["source_assets"] = copy.deepcopy(duplicate["source_assets"])
    source["case_records"].append(duplicate)
    source["case_count"] += 1

    with pytest.raises(DevelopmentManifestError, match="duplicate case_id"):
        _compile(bundle)


def test_rejects_too_few_unique_patients() -> None:
    bundle = _source_bundle(patient_count=9)

    with pytest.raises(DevelopmentManifestError, match="at least 10"):
        _compile(bundle)


def test_publish_is_no_clobber_and_roots_must_be_physically_disjoint(
    tmp_path: Path,
) -> None:
    source_path, audit_receipt_path, split_path = _write_inputs(
        tmp_path,
        _source_bundle(),
    )
    visible = tmp_path / "visible"
    evaluation = tmp_path / "evaluation"

    build_development_manifests(
        source_path,
        audit_receipt_path=audit_receipt_path,
        patient_split_path=split_path,
        visible_root=visible,
        eval_root=evaluation,
    )
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        build_development_manifests(
            source_path,
            audit_receipt_path=audit_receipt_path,
            patient_split_path=split_path,
            visible_root=visible,
            eval_root=evaluation,
        )
    with pytest.raises(DevelopmentManifestError, match="physically disjoint"):
        build_development_manifests(
            source_path,
            audit_receipt_path=audit_receipt_path,
            patient_split_path=split_path,
            visible_root=tmp_path / "nested",
            eval_root=tmp_path / "nested" / "eval",
        )


def test_document_validator_rejects_duplicate_patient_and_public_leak() -> None:
    bundle = _source_bundle()
    artifacts = _compile(bundle)
    duplicate = copy.deepcopy(artifacts)
    duplicate["private"]["patients"][1]["patient_id"] = duplicate["private"][
        "patients"
    ][0]["patient_id"]
    leaked = copy.deepcopy(artifacts)
    leaked["public"]["patients"][0]["case_id"] = "secret-case"

    with pytest.raises(DevelopmentManifestError, match="duplicate patient"):
        validate_development_manifest_documents(
            duplicate["public"],
            duplicate["private"],
        )
    with pytest.raises(DevelopmentManifestError, match="public patient fields"):
        validate_development_manifest_documents(
            leaked["public"],
            leaked["private"],
        )


def test_published_validator_detects_tampering(tmp_path: Path) -> None:
    source_path, audit_receipt_path, split_path = _write_inputs(
        tmp_path,
        _source_bundle(),
    )
    receipt = build_development_manifests(
        source_path,
        audit_receipt_path=audit_receipt_path,
        patient_split_path=split_path,
        visible_root=tmp_path / "visible",
        eval_root=tmp_path / "eval",
    )
    private_path = Path(receipt["private_manifest_path"])
    private = json.loads(private_path.read_text(encoding="utf-8"))
    private["patients"][0]["scribble_strategy"] = "random"
    private_path.write_text(
        json.dumps(private, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(DevelopmentManifestError, match="hash mismatch"):
        validate_published_development_manifests(
            source_path=source_path,
            audit_receipt_path=audit_receipt_path,
            patient_split_path=split_path,
            public_manifest_path=Path(receipt["public_manifest_path"]),
            private_manifest_path=private_path,
            receipt_path=Path(receipt["receipt_path"]),
        )
