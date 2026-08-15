from __future__ import annotations

import inspect
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest


PROJECT = Path(__file__).resolve().parents[1]
SCRIPTS = PROJECT / "scripts"
sys.path.insert(0, str(SCRIPTS))

from common import petct_m0_test_access as access  # noqa: E402
from common.petct_m0_test_access import (  # noqa: E402
    M0_TEST_CONFIRMATION,
    PARTITION,
    SCOPE,
    consume_m0_test_grant,
    create_m0_test_grant,
    enforce_m0_test_access,
    validate_m0_test_receipt,
)
from common.petct_test_access import (  # noqa: E402
    RECEIPT_SCHEMA as FINAL_RECEIPT_SCHEMA,
    TestAccessError,
    _canonical_sha256,
    _sealed,
    validate_consumed_receipt,
)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _canonical_ledger(tmp_path: Path) -> Path:
    return (
        tmp_path
        / "records"
        / "test-access"
        / "m0-baseline-global-consumption-ledger"
    )


def _build_contracts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    partition_specs = (
        ("train", 264, 151),
        ("val", 57, 34),
        ("test", 57, 34),
    )
    identity_rows: list[dict[str, Any]] = []
    patient_rows: list[dict[str, Any]] = []
    case_counts: dict[str, int] = {}
    patient_index = 0
    case_index = 0
    for partition, patient_count, patients_with_second_case in partition_specs:
        partition_cases = 0
        if partition == "test":
            # Frozen real inventory: cases per held-out fold = 20/14/23/19/15.
            # With 57 patients and 34 second examinations, one valid patient-level
            # allocation is (patients, doubles) below.
            patient_layout = [
                (fold, 2 if local_patient < double_count else 1)
                for fold, fold_patients, double_count in (
                    (0, 12, 8),
                    (1, 8, 6),
                    (2, 13, 10),
                    (3, 12, 7),
                    (4, 12, 3),
                )
                for local_patient in range(fold_patients)
            ]
            assert len(patient_layout) == patient_count
        else:
            patient_layout = [
                (
                    (patient_index + local_patient) % 5,
                    2 if local_patient < patients_with_second_case else 1,
                )
                for local_patient in range(patient_count)
            ]
        for fold, number_of_cases in patient_layout:
            patient_id = f"patient-{patient_index:04d}"
            case_ids: list[str] = []
            for _ in range(number_of_cases):
                case_id = f"case-{case_index:04d}"
                case_ids.append(case_id)
                leaf_root = tmp_path / "unopened" / case_id
                identity_rows.append(
                    {
                        "case_id": case_id,
                        "patient_id": patient_id,
                        "held_out_fold": fold,
                        "ct_path": str((leaf_root / "ct.nii.gz").resolve()),
                        "pet_path": str((leaf_root / "pet.nii.gz").resolve()),
                        "gt_path": str((leaf_root / "gt.nii.gz").resolve()),
                        "truth_materialization": "IDENTITY_ONLY",
                    }
                )
                case_index += 1
                partition_cases += 1
            patient_rows.append(
                {
                    "patient_id": patient_id,
                    "partition": partition,
                    "case_ids": case_ids,
                }
            )
            patient_index += 1
        case_counts[partition] = partition_cases

    assert len(identity_rows) == 597
    assert patient_index == 378
    assert case_counts == {"train": 415, "val": 91, "test": 91}
    identity_rows.sort(key=lambda row: row["case_id"])
    identity_manifest = tmp_path / "identity.jsonl"
    identity_manifest.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in identity_rows
        ),
        encoding="utf-8",
    )

    config = tmp_path / "experiment.json"
    _write_json(
        config,
        {
            "schema_version": "PETCT-ROUTE-A-EXPERIMENT-v2.0",
            "dataset": {
                "name": "PSMA-PET-CT-Lesions-v3",
                "cases": 597,
                "patients": 378,
                "split_unit": "patient",
                "folds": 5,
                "learning_split": {
                    "schema_version": "PETCT-LEARNING-SPLIT-v1.0",
                    "algorithm": "stable-patient-hash-v1",
                    "seed": 20260717,
                    "target_patient_counts": {
                        "train": 264,
                        "val": 57,
                        "test": 57,
                    },
                    "case_counts": "FROZEN_IN_GENERATED_SPLIT_RECEIPT",
                    "test_access": (
                        "exactly-once-after-all-v2-development-freezes"
                    ),
                },
            },
            "m0": {
                "role": (
                    "one OOF family: every case receives exactly one "
                    "held-out-fold prediction"
                )
            },
            "statistics": {"test_access": "exactly-once-after-v2-freeze"},
        },
    )
    learning_split = tmp_path / "learning-split.json"
    _write_json(
        learning_split,
        {
            "schema_version": "PETCT-LEARNING-SPLIT-v1.0",
            "status": "FROZEN_BEFORE_MODEL_SELECTION",
            "dataset": "PSMA-PET-CT-Lesions-v3",
            "split_unit": "patient",
            "case_count": 597,
            "patient_count": 378,
            "algorithm": "stable-patient-hash-v1",
            "seed": 20260717,
            "target_patient_counts": {"train": 264, "val": 57, "test": 57},
            "case_counts": case_counts,
            "patients": patient_rows,
        },
    )

    oof_ready = tmp_path / "nnunet" / "manifests" / "OOF_READY.json"
    _write_json(oof_ready, {"fixture": True})
    oof_cases = {
        row["case_id"]: {
            "case_id": row["case_id"],
            "patient_id": row["patient_id"],
            "held_out_fold": row["held_out_fold"],
        }
        for row in identity_rows
    }

    def validate_ready(path: Path) -> dict[str, Any]:
        resolved = path.resolve()
        return {
            "status": "PASS",
            "schema_version": "PETCT-M0-OOF-READY-v1.0",
            "phase": "PATIENT_EXCLUDED_5FOLD_OOF",
            "ready_path": str(resolved),
            "ready_sha256": access._file_record(
                resolved, label="fixture OOF_READY"
            )["sha256"],
            "patient_excluded": True,
            "cases": oof_cases,
        }

    monkeypatch.setattr(access, "validate_oof_ready_receipt_only", validate_ready)
    official_metrics = tmp_path / "official_metrics.py"
    evaluator_script = tmp_path / "evaluate_petct_m0_oof.py"
    runner_script = tmp_path / "run_petct_m0_test_baseline.py"
    official_metrics.write_text("# pinned official metrics\n", encoding="utf-8")
    evaluator_script.write_text("# pinned M0 evaluator\n", encoding="utf-8")
    runner_script.write_text("# pinned M0 runner\n", encoding="utf-8")
    run_root = tmp_path / "run"
    run_root.mkdir()
    outputs = (run_root / "m0_test_summary.json",)
    return {
        "experiment_config": config,
        "learning_split": learning_split,
        "identity_manifest": identity_manifest,
        "oof_ready": oof_ready,
        "official_metrics": official_metrics,
        "evaluator_script": evaluator_script,
        "runner_script": runner_script,
        "run_root": run_root,
        "output_paths": outputs,
        "oof_cases": oof_cases,
    }


def _grant_and_consume(
    fixture: dict[str, Any], tmp_path: Path
) -> tuple[Path, Path, Path, dict[str, Any]]:
    grant = tmp_path / "M0_TEST_GRANT.json"
    create_m0_test_grant(
        **{
            key: fixture[key]
            for key in (
                "experiment_config",
                "learning_split",
                "identity_manifest",
                "oof_ready",
                "official_metrics",
                "evaluator_script",
                "runner_script",
                "run_root",
                "output_paths",
            )
        },
        grant_path=grant,
        authorized_by="codex-on-director-authority",
        confirmation=M0_TEST_CONFIRMATION,
        now=lambda: "2026-08-01T00:00:00Z",
    )
    receipt = fixture["run_root"] / "governance" / "M0_TEST_CONSUMED.json"
    ledger = _canonical_ledger(tmp_path)
    payload = consume_m0_test_grant(
        grant_path=grant,
        run_root=fixture["run_root"],
        output_paths=fixture["output_paths"],
        receipt_path=receipt,
        ledger_root=ledger,
        now=lambda: "2026-08-01T00:01:00Z",
    )
    return grant, receipt, ledger, payload


def _enforce(
    fixture: dict[str, Any], receipt: Path, ledger: Path, **overrides: Any
) -> dict[str, Any]:
    arguments = {
        key: fixture[key]
        for key in (
            "experiment_config",
            "learning_split",
            "identity_manifest",
            "oof_ready",
            "official_metrics",
            "evaluator_script",
            "runner_script",
            "run_root",
            "output_paths",
        )
    }
    arguments.update(overrides)
    return enforce_m0_test_access(
        receipt_path=receipt, ledger_root=ledger, **arguments
    )


def test_enforcer_is_keyword_only_and_has_the_frozen_interface() -> None:
    signature = inspect.signature(enforce_m0_test_access)
    expected = {
        "receipt_path",
        "experiment_config",
        "learning_split",
        "identity_manifest",
        "oof_ready",
        "official_metrics",
        "evaluator_script",
        "runner_script",
        "run_root",
        "output_paths",
        "ledger_root",
    }
    assert set(signature.parameters) == expected
    assert all(
        parameter.kind is inspect.Parameter.KEYWORD_ONLY
        for parameter in signature.parameters.values()
    )


def test_cli_grant_consume_validate_use_an_explicit_external_ledger(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fixture = _build_contracts(tmp_path, monkeypatch)
    grant = tmp_path / "cli-grant.json"
    receipt = fixture["run_root"] / "governance" / "cli-receipt.json"
    ledger = _canonical_ledger(tmp_path)
    binding_args = [
        "--experiment-config",
        str(fixture["experiment_config"]),
        "--learning-split",
        str(fixture["learning_split"]),
        "--identity-manifest",
        str(fixture["identity_manifest"]),
        "--oof-ready",
        str(fixture["oof_ready"]),
        "--official-metrics",
        str(fixture["official_metrics"]),
        "--evaluator-script",
        str(fixture["evaluator_script"]),
        "--runner-script",
        str(fixture["runner_script"]),
        "--run-root",
        str(fixture["run_root"]),
    ]
    for output in fixture["output_paths"]:
        binding_args.extend(("--output-path", str(output)))

    assert access.main(
        [
            "grant",
            *binding_args,
            "--grant",
            str(grant),
            "--authorized-by",
            "codex-on-director-authority",
            "--confirm-m0-test",
            M0_TEST_CONFIRMATION,
        ]
    ) == 0
    consume_args = [
        "consume",
        "--grant",
        str(grant),
        "--run-root",
        str(fixture["run_root"]),
    ]
    for output in fixture["output_paths"]:
        consume_args.extend(("--output-path", str(output)))
    consume_args.extend(
        (
            "--receipt",
            str(receipt),
            "--ledger-root",
            str(ledger),
        )
    )
    assert access.main(consume_args) == 0
    assert access.main(
        [
            "validate",
            *binding_args,
            "--receipt",
            str(receipt),
            "--ledger-root",
            str(ledger),
        ]
    ) == 0
    capsys.readouterr()

    assert ledger.resolve() != fixture["run_root"].resolve()
    assert len(list(ledger.glob("*.json"))) == 1


def test_consumed_receipt_binds_exact_m0_test_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _build_contracts(tmp_path, monkeypatch)
    _, receipt, ledger, payload = _grant_and_consume(fixture, tmp_path)

    validated = _enforce(fixture, receipt, ledger)

    assert validated["receipt_sha256"] == payload["receipt_sha256"]
    assert len(validated["receipt_sha256"]) == 64
    assert validated["scope"] == SCOPE
    assert validated["allowed_partition"] == PARTITION
    binding = validated["consumption"]["binding"]
    assert binding["test_inventory"]["case_count"] == 91
    assert binding["test_inventory"]["patient_count"] == 57
    assert binding["test_inventory"]["held_out_fold_case_counts"] == {
        "0": 20,
        "1": 14,
        "2": 23,
        "3": 19,
        "4": 15,
    }
    assert binding["five_fold_ensemble"] is False
    assert binding["m0_evaluator_script"]["sha256"]
    assert binding["m0_runner_script"]["sha256"]


def test_existing_outputs_do_not_invalidate_a_consumed_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _build_contracts(tmp_path, monkeypatch)
    _, receipt, ledger, _ = _grant_and_consume(fixture, tmp_path)
    for output in fixture["output_paths"]:
        output.write_text("published\n", encoding="utf-8")

    assert _enforce(fixture, receipt, ledger)["receipt_sha256"]


def test_wrong_confirmation_fails_before_grant_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _build_contracts(tmp_path, monkeypatch)
    grant = tmp_path / "grant.json"
    with pytest.raises(TestAccessError, match="confirmation phrase"):
        create_m0_test_grant(
            **{
                key: fixture[key]
                for key in (
                    "experiment_config",
                    "learning_split",
                    "identity_manifest",
                    "oof_ready",
                    "official_metrics",
                    "evaluator_script",
                    "runner_script",
                    "run_root",
                    "output_paths",
                )
            },
            grant_path=grant,
            authorized_by="director",
            confirmation="yes",
        )
    assert not grant.exists()


def test_wrong_experiment_schema_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _build_contracts(tmp_path, monkeypatch)
    config = json.loads(fixture["experiment_config"].read_text(encoding="utf-8"))
    config["schema_version"] = "PETCT-ROUTE-A-EXPERIMENT-v1.0"
    config["dataset"]["learning_split"]["test_access"] = (
        "exactly-once-after-all-development-freezes"
    )
    config["statistics"]["test_access"] = "exactly-once-after-freeze"
    _write_json(fixture["experiment_config"], config)
    with pytest.raises(TestAccessError, match="requires the v2 experiment schema"):
        create_m0_test_grant(
            **{
                key: fixture[key]
                for key in (
                    "experiment_config",
                    "learning_split",
                    "identity_manifest",
                    "oof_ready",
                    "official_metrics",
                    "evaluator_script",
                    "runner_script",
                    "run_root",
                    "output_paths",
                )
            },
            grant_path=tmp_path / "grant.json",
            authorized_by="director",
            confirmation=M0_TEST_CONFIRMATION,
        )


def test_oof_case_identity_mismatch_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _build_contracts(tmp_path, monkeypatch)
    fixture["oof_cases"]["case-0506"]["held_out_fold"] = 4
    with pytest.raises(TestAccessError, match="case/patient/fold identity differs"):
        create_m0_test_grant(
            **{
                key: fixture[key]
                for key in (
                    "experiment_config",
                    "learning_split",
                    "identity_manifest",
                    "oof_ready",
                    "official_metrics",
                    "evaluator_script",
                    "runner_script",
                    "run_root",
                    "output_paths",
                )
            },
            grant_path=tmp_path / "grant.json",
            authorized_by="director",
            confirmation=M0_TEST_CONFIRMATION,
        )


def test_test_fold_inventory_drift_is_rejected_before_grant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _build_contracts(tmp_path, monkeypatch)
    rows = [
        json.loads(line)
        for line in fixture["identity_manifest"]
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    patient_counts: dict[str, int] = {}
    for row in rows:
        patient_counts[row["patient_id"]] = patient_counts.get(row["patient_id"], 0) + 1
    changed = next(
        row
        for row in rows
        if row["case_id"] >= "case-0506"
        and row["held_out_fold"] == 0
        and patient_counts[row["patient_id"]] == 1
    )
    changed["held_out_fold"] = 1
    fixture["identity_manifest"].write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    fixture["oof_cases"][changed["case_id"]]["held_out_fold"] = 1

    with pytest.raises(TestAccessError, match="20/14/23/19/15"):
        create_m0_test_grant(
            **{
                key: fixture[key]
                for key in (
                    "experiment_config",
                    "learning_split",
                    "identity_manifest",
                    "oof_ready",
                    "official_metrics",
                    "evaluator_script",
                    "runner_script",
                    "run_root",
                    "output_paths",
                )
            },
            grant_path=tmp_path / "fold-drift-grant.json",
            authorized_by="director",
            confirmation=M0_TEST_CONFIRMATION,
        )


def test_output_escape_and_output_rebinding_are_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _build_contracts(tmp_path, monkeypatch)
    escaped = (*fixture["output_paths"][:-1], tmp_path / "outside.json")
    with pytest.raises(TestAccessError, match="escapes"):
        create_m0_test_grant(
            **{
                key: fixture[key]
                for key in (
                    "experiment_config",
                    "learning_split",
                    "identity_manifest",
                    "oof_ready",
                    "official_metrics",
                    "evaluator_script",
                    "runner_script",
                    "run_root",
                )
            },
            output_paths=escaped,
            grant_path=tmp_path / "escape-grant.json",
            authorized_by="director",
            confirmation=M0_TEST_CONFIRMATION,
        )
    _, receipt, ledger, _ = _grant_and_consume(fixture, tmp_path)
    changed = (*fixture["output_paths"][:-1], fixture["run_root"] / "other.json")
    with pytest.raises(TestAccessError, match="run/output binding differs"):
        _enforce(fixture, receipt, ledger, output_paths=changed)


def test_exactly_one_concurrent_global_consumption_wins(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _build_contracts(tmp_path, monkeypatch)
    grant = tmp_path / "grant.json"
    create_m0_test_grant(
        **{
            key: fixture[key]
            for key in (
                "experiment_config",
                "learning_split",
                "identity_manifest",
                "oof_ready",
                "official_metrics",
                "evaluator_script",
                "runner_script",
                "run_root",
                "output_paths",
            )
        },
        grant_path=grant,
        authorized_by="director",
        confirmation=M0_TEST_CONFIRMATION,
    )
    ledger = _canonical_ledger(tmp_path)
    receipt = fixture["run_root"] / "receipt.json"

    def consume() -> dict[str, Any] | TestAccessError:
        try:
            return consume_m0_test_grant(
                grant_path=grant,
                run_root=fixture["run_root"],
                output_paths=fixture["output_paths"],
                receipt_path=receipt,
                ledger_root=ledger,
            )
        except TestAccessError as exc:
            return exc

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: consume(), range(2)))
    assert sum(isinstance(result, dict) for result in results) == 1
    assert sum(isinstance(result, TestAccessError) for result in results) == 1
    assert len(list(ledger.glob("*.json"))) == 1
    with pytest.raises(TestAccessError, match="already consumed globally"):
        consume_m0_test_grant(
            grant_path=grant,
            run_root=fixture["run_root"],
            output_paths=fixture["output_paths"],
            receipt_path=fixture["run_root"] / "replay.json",
            ledger_root=ledger,
        )


def test_alternate_ledger_root_cannot_reset_or_validate_global_consumption(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _build_contracts(tmp_path, monkeypatch)
    grant, receipt, canonical_ledger, _ = _grant_and_consume(fixture, tmp_path)
    alternate = tmp_path / "alternate-ledger"

    with pytest.raises(TestAccessError, match="not the canonical project ledger"):
        consume_m0_test_grant(
            grant_path=grant,
            run_root=fixture["run_root"],
            output_paths=fixture["output_paths"],
            receipt_path=fixture["run_root"] / "alternate-receipt.json",
            ledger_root=alternate,
        )
    assert not alternate.exists()

    def forbidden_scientific_read(**_: Any) -> dict[str, Any]:
        raise AssertionError("scientific binding read before canonical-ledger check")

    monkeypatch.setattr(access, "_build_binding", forbidden_scientific_read)
    with pytest.raises(TestAccessError, match="not the canonical project ledger"):
        _enforce(fixture, receipt, alternate)
    assert len(list(canonical_ledger.glob("*.json"))) == 1


def test_code_or_scientific_input_mutation_invalidates_grant_and_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _build_contracts(tmp_path, monkeypatch)
    grant, receipt, ledger, _ = _grant_and_consume(fixture, tmp_path)
    fixture["runner_script"].write_text("# changed runner\n", encoding="utf-8")

    with pytest.raises(TestAccessError, match="binding changed"):
        _enforce(fixture, receipt, ledger)
    with pytest.raises(TestAccessError, match="binding changed after grant"):
        consume_m0_test_grant(
            grant_path=grant,
            run_root=fixture["run_root"],
            output_paths=fixture["output_paths"],
            receipt_path=fixture["run_root"] / "other.json",
            ledger_root=tmp_path / "other-ledger",
        )


def test_missing_global_ledger_fails_before_scientific_binding_io(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _build_contracts(tmp_path, monkeypatch)
    _, receipt, ledger, _ = _grant_and_consume(fixture, tmp_path)
    ledger_records = list(ledger.glob("*.json"))
    assert len(ledger_records) == 1
    ledger_records[0].unlink()

    def forbidden_scientific_read(**_: Any) -> dict[str, Any]:
        raise AssertionError("scientific binding read before ledger validation")

    monkeypatch.setattr(access, "_build_binding", forbidden_scientific_read)
    with pytest.raises(TestAccessError, match="global consumption ledger"):
        _enforce(fixture, receipt, ledger)


@pytest.mark.parametrize(
    ("field", "value"),
    (("scope", "p2t"), ("allowed_partition", "val")),
)
def test_resealed_non_m0_or_non_test_receipt_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: str,
) -> None:
    fixture = _build_contracts(tmp_path, monkeypatch)
    _, receipt, ledger, _ = _grant_and_consume(fixture, tmp_path)
    forged = json.loads(receipt.read_text(encoding="utf-8"))
    forged["consumption"][field] = value
    forged["consumption_sha256"] = _canonical_sha256(forged["consumption"])
    forged = _sealed(forged, "receipt_sha256")
    _write_json(receipt, forged)

    with pytest.raises(TestAccessError, match="not test-only baseline scope"):
        _enforce(fixture, receipt, ledger)


def test_final_and_m0_receipts_are_mutually_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _build_contracts(tmp_path, monkeypatch)
    _, m0_receipt, ledger, _ = _grant_and_consume(fixture, tmp_path)
    with pytest.raises(TestAccessError, match="contract is invalid"):
        validate_consumed_receipt(
            m0_receipt,
            experiment_config=fixture["experiment_config"],
            learning_split=fixture["learning_split"],
            run_root=fixture["run_root"],
            ledger_root=ledger,
        )

    final_receipt = fixture["run_root"] / "fake-final-receipt.json"
    _write_json(
        final_receipt,
        _sealed(
            {
                "schema_version": FINAL_RECEIPT_SCHEMA,
                "status": "CONSUMED",
                "consumption": {},
                "consumption_sha256": _canonical_sha256({}),
                "global_ledger": {},
            },
            "receipt_sha256",
        ),
    )
    with pytest.raises(TestAccessError, match="contract is invalid"):
        validate_m0_test_receipt(
            final_receipt,
            **{
                key: fixture[key]
                for key in (
                    "experiment_config",
                    "learning_split",
                    "identity_manifest",
                    "oof_ready",
                    "official_metrics",
                    "evaluator_script",
                    "runner_script",
                    "run_root",
                    "output_paths",
                )
            },
            ledger_root=ledger,
        )
