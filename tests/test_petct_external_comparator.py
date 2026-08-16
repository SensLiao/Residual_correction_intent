from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest


PROJECT = Path(__file__).resolve().parents[1]
SCRIPTS = PROJECT / "scripts"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(SCRIPTS / "comparators"))

from run_petct_external_comparator import (  # noqa: E402
    ContractError,
    build_execution_plan,
    load_and_validate_contract,
    main,
    validate_contract,
    validate_execution_admission,
    validate_manifest,
)


CONFIG = PROJECT / "configs" / "petct_external_comparators.json"
REQUIRED_METHODS = {
    "autopetv_official_4ch_nnunet",
    "scribbleprompt",
    "sw_fastedit",
    "prism",
    "nninteractive",
}


def _source_license_or_skip(
    source_id: str,
    configured_path: Path,
    minimal_runtime_path: Path | None,
) -> Path:
    for candidate in (configured_path, minimal_runtime_path):
        if candidate is not None and candidate.parent.is_dir():
            assert candidate.is_file(), (
                f"{source_id} source bundle is present but its license is missing: "
                f"{candidate}"
            )
            return candidate
    pytest.skip(
        f"requires the pinned {source_id} source checkout or minimal runtime "
        "license; neither vendor asset directory is present"
    )


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _wire_test_admission(
    method: dict[str, object], tmp_path: Path
) -> tuple[Path, dict[str, Path]]:
    receipt_path = tmp_path / "runtime.READY.json"
    files = {
        "adapter_sha256": tmp_path / "adapter.py",
        "checkpoint_sha256": tmp_path / "checkpoint.pth",
        "license_sha256": tmp_path / "MODEL-LICENSE",
        "environment_freeze_sha256": tmp_path / "environment.freeze.txt",
        "source_license_sha256": tmp_path / "SOURCE-LICENSE",
    }
    for index, path in enumerate(files.values()):
        path.write_bytes(f"bound-file-{index}".encode("utf-8"))
    execution = method["execution"]
    assert isinstance(execution, dict)
    execution["admission"] = {
        "receipt": str(receipt_path),
        "schema_version": "PETCT-TEST-ENV-v1.0",
        "status": "PASS",
        "config_sha256_field": "config_sha256",
        "required_pass_fields": ["model_load_smoke", "adapter_cli_smoke"],
        "exact_fields": {
            "synthetic_only": True,
            "scientific_prediction_produced": False,
            "network_policy_at_runtime": "NO_DOWNLOADS",
        },
        "file_sha256_fields": {key: str(path) for key, path in files.items()},
    }
    return receipt_path, files


def _write_test_admission_receipt(
    receipt_path: Path,
    config: Path,
    files: dict[str, Path],
) -> None:
    _write_json(
        receipt_path,
        {
            "schema_version": "PETCT-TEST-ENV-v1.0",
            "status": "PASS",
            "config_sha256": _sha256(config),
            "model_load_smoke": "PASS",
            "adapter_cli_smoke": "PASS",
            "synthetic_only": True,
            "scientific_prediction_produced": False,
            "network_policy_at_runtime": "NO_DOWNLOADS",
            **{field: _sha256(path) for field, path in files.items()},
        },
    )


def _sync_test_admission_record(contract: dict[str, object], method: dict[str, object]) -> None:
    review = contract["method_selection_review"]
    assert isinstance(review, dict)
    register = review["machine_readable_admission_register"]
    assert isinstance(register, dict)
    records = register["records"]
    assert isinstance(records, list)
    record = next(item for item in records if item["id"] == method["id"])
    execution = method["execution"]
    assert isinstance(execution, dict)
    record["execution_state"] = execution["state"]


def test_contract_declares_required_external_comparator_classes() -> None:
    contract = load_and_validate_contract(CONFIG)
    methods = {method["id"]: method for method in contract["methods"]}

    assert REQUIRED_METHODS <= methods.keys()
    assert methods["autopetv_official_4ch_nnunet"]["spatial_dimensionality"] == "3D"
    assert methods["scribbleprompt"]["spatial_dimensionality"] == "2D"
    assert "foreground_click" in methods["sw_fastedit"]["prompt_modalities"]
    assert "scribble" in methods["prism"]["prompt_modalities"]
    assert "scribble" in methods["nninteractive"]["prompt_modalities"]


def test_selected_source_licenses_are_machine_readable_and_distinct_from_checkpoint_licenses() -> None:
    contract = load_and_validate_contract(CONFIG)
    methods = {method["id"]: method for method in contract["methods"]}
    protocols = {
        protocol["id"]: protocol for protocol in contract["protocol_adapters"]
    }
    expected = {
        "autopetv_scribble_simulator": (
            protocols["autopetv_scribble_simulator"]["source"],
            "1eb85fc97224598dad1852b5d6483bbcf0aa8608790dcc657a5a2a761ae9c8c6",
            PROJECT / "external_runners" / "autopetv_protocol" / "LICENSE",
        ),
        "scribbleprompt": (
            methods["scribbleprompt"]["source"],
            "1eb85fc97224598dad1852b5d6483bbcf0aa8608790dcc657a5a2a761ae9c8c6",
            None,
        ),
        "nninteractive": (
            methods["nninteractive"]["source"],
            "3888a43f438f1834561474a1b16cfd0b11037c7a225935a8b89558aac550167e",
            PROJECT / "external_runners" / "nninteractive" / "source" / "LICENSE",
        ),
    }
    for source_id, (source, expected_sha256, minimal_runtime_license) in expected.items():
        assert source["license"] == "Apache-2.0"
        assert source["license_sha256"] == expected_sha256
        license_path = PROJECT / source["license_file"]
        if source_id == "scribbleprompt" and not license_path.is_file():
            assert methods["scribbleprompt"]["admission_state"] == "BLOCKED_CHECKPOINT_LICENSE"
            continue
        license_path = _source_license_or_skip(
            source_id, license_path, minimal_runtime_license
        )
        assert hashlib.sha256(license_path.read_bytes()).hexdigest() == expected_sha256

    scribbleprompt = methods["scribbleprompt"]
    assert (
        scribbleprompt["pretraining"]["checkpoint_license"]
        == "UNRESOLVED_NO_INDEPENDENT_WEIGHT_LICENSE_FOUND"
    )
    nninteractive_model = methods["nninteractive"]["pretraining"][
        "local_checkpoint_availability"
    ]
    assert nninteractive_model["license"] == "CC BY-NC-SA 4.0"
    assert (
        nninteractive_model["license_sha256"]
        != methods["nninteractive"]["source"]["license_sha256"]
    )


def test_conditioning_module_references_bind_exact_papers_sources_and_limits() -> None:
    contract = load_and_validate_contract(CONFIG)
    references = {
        reference["id"]: reference
        for reference in contract["conditioning_module_references"]
    }
    assert references.keys() == {"film", "lvit"}

    source_licenses = [
        PROJECT / reference["source"]["license_file"]
        for reference in references.values()
    ]
    missing_source_bundles = [
        path.parent for path in source_licenses if not path.parent.is_dir()
    ]
    if missing_source_bundles:
        pytest.skip(
            "requires the pinned FiLM and LViT source-license bundles for "
            "paper/source hash verification; absent directories: "
            + ", ".join(str(path) for path in missing_source_bundles)
        )
    for reference in references.values():
        assert reference["server_artifact_required"] is False
        paper_path = (PROJECT / reference["paper"]["path"]).resolve()
        source_license = PROJECT / reference["source"]["license_file"]
        assert paper_path.is_file()
        assert source_license.is_file()
        assert hashlib.sha256(paper_path.read_bytes()).hexdigest() == reference["paper"]["sha256"]
        assert (
            hashlib.sha256(source_license.read_bytes()).hexdigest()
            == reference["source"]["license_sha256"]
        )

    assert references["film"]["source"]["license"] == "CC BY-NC 4.0"
    assert references["lvit"]["source"]["license"] == "MIT"
    assert "first DownViT" in references["lvit"]["paper_code_contract"]
    assert "no text-image cross-attention" in references["lvit"]["paper_code_contract"]

    fusion_sources = {
        source["id"]: source
        for source in contract["petct_fusion_source_audit"]["sources"]
    }
    aatsn = fusion_sources["aatsn"]
    assert aatsn["paper_access"] == "PENDING_LEGAL_FULLTEXT_ABSTRACT_ONLY"
    assert aatsn["repository"] is None
    assert aatsn["license"] == "NOT_FOUND"


def test_every_method_declares_same_manifests_metrics_and_leakage_state() -> None:
    contract = load_and_validate_contract(CONFIG)
    expected_metrics = contract["metrics_contract"]["required_metrics"]

    for method in contract["methods"]:
        assert method["contracts"] == {
            "input_manifest": contract["input_manifest_contract"]["schema_version"],
            "output_manifest": contract["output_manifest_contract"]["schema_version"],
            "metrics": contract["metrics_contract"]["schema_version"],
        }
        assert method["metric_set"] == expected_metrics
        assert method["pretraining"]["current_psma_v3_exposure"] in {
            "KNOWN_PUBLIC_COHORT_EXPOSURE",
            "UNVERIFIED",
            "NOT_APPLICABLE_FROM_SCRATCH",
        }
        assert isinstance(method["headline"]["eligible"], bool)
        assert method["headline"]["reason"]
        assert method["execution"]["network_policy"] == "NO_DOWNLOADS"


def test_machine_admission_register_and_role_invariants_are_fail_closed() -> None:
    contract = load_and_validate_contract(CONFIG)
    register = contract["method_selection_review"][
        "machine_readable_admission_register"
    ]
    assert register["schema_version"] == "PETCT-METHOD-ADMISSION-REGISTER-v1.0"
    assert "typed records" in register["record_source"]
    assert "current_config_runtime_receipt" in register["required_evidence"]
    records = {record["id"]: record for record in register["records"]}
    assert records["autopetv_scribble_simulator"]["role"] == "ADAPT"
    assert records["nninteractive"]["execution_state"] == "ARGV_WIRED"
    assert {
        blocker["id"] for blocker in records["nninteractive"]["blockers"]
    } == {"patient_excluded_oof_inputs", "current_config_runtime_receipt"}
    assert records["scribbleprompt"]["server_package_status"] == "NOT_UPLOADED"
    assert records["prism"]["role"] == "REFERENCE_ONLY"

    missing_record = copy.deepcopy(contract)
    missing_record["method_selection_review"]["machine_readable_admission_register"][
        "records"
    ] = [
        record
        for record in register["records"]
        if record["id"] != "nninteractive"
    ]
    with pytest.raises(ContractError, match="omits classified id nninteractive"):
        validate_contract(missing_record)

    stale_role = copy.deepcopy(contract)
    stale_records = stale_role["method_selection_review"][
        "machine_readable_admission_register"
    ]["records"]
    next(record for record in stale_records if record["id"] == "prism")["role"] = "RUN"
    with pytest.raises(ContractError, match="role mismatch for prism"):
        validate_contract(stale_role)

    reference = copy.deepcopy(contract)
    reference_method = next(
        method for method in reference["methods"] if method["id"] == "prism"
    )
    reference_method["execution"] = copy.deepcopy(
        next(
            method
            for method in contract["methods"]
            if method["id"] == "nninteractive"
        )["execution"]
    )
    with pytest.raises(ContractError, match="REFERENCE_ONLY"):
        validate_contract(reference)

    unlicensed = copy.deepcopy(contract)
    scribbleprompt = next(
        method for method in unlicensed["methods"] if method["id"] == "scribbleprompt"
    )
    scribbleprompt["execution"] = copy.deepcopy(
        next(
            method
            for method in contract["methods"]
            if method["id"] == "nninteractive"
        )["execution"]
    )
    with pytest.raises(ContractError, match="checkpoint license is unresolved"):
        validate_contract(unlicensed)


def test_default_cli_is_dry_run_and_never_calls_subprocess(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("dry-run must never call subprocess.run")

    monkeypatch.setattr(subprocess, "run", forbidden)
    rc = main(
        [
            "--config",
            str(CONFIG),
            "--method",
            "scribbleprompt",
            "--input-manifest",
            "planned-input.json",
            "--output-manifest",
            "planned-output.json",
        ]
    )
    printed = json.loads(capsys.readouterr().out)

    assert rc == 0
    assert printed["mode"] == "DRY_RUN"
    assert printed["would_execute"] is False
    assert printed["execution_state"] == "NOT_WIRED"


def test_execute_without_exact_confirmation_is_rejected_before_subprocess(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called = False

    def forbidden(*args: object, **kwargs: object) -> None:
        nonlocal called
        called = True

    monkeypatch.setattr(subprocess, "run", forbidden)
    with pytest.raises(ContractError, match="confirmation token"):
        main(
            [
                "--config",
                str(CONFIG),
                "--method",
                "scribbleprompt",
                "--input-manifest",
                "input.json",
                "--output-manifest",
                "output.json",
                "--execute",
            ]
        )
    assert called is False


def test_generic_runner_test_execution_is_formal_launcher_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called = False

    def forbidden(*args: object, **kwargs: object) -> None:
        nonlocal called
        called = True

    monkeypatch.setattr(subprocess, "run", forbidden)
    contract = load_and_validate_contract(CONFIG)
    with pytest.raises(ContractError, match="formal external comparator launcher"):
        main(
            [
                "--config",
                str(CONFIG),
                "--method",
                "nninteractive",
                "--input-manifest",
                str(tmp_path / "must-not-be-read.json"),
                "--output-manifest",
                str(tmp_path / "must-not-exist.json"),
                "--learning-split",
                str(tmp_path / "must-not-be-read-split.json"),
                "--partition",
                "test",
                "--execute",
                "--confirm",
                contract["execution_policy"]["confirmation_token"],
            ]
        )
    assert called is False


def test_execute_uses_argv_without_shell_and_with_offline_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = load_and_validate_contract(CONFIG)
    modified = copy.deepcopy(contract)
    method = next(item for item in modified["methods"] if item["id"] == "scribbleprompt")
    method["pretraining"]["checkpoint_license"] = "UNIT_TEST_LICENSE_EVIDENCE"
    method["execution"] = {
        "state": "ARGV_WIRED",
        "argv": [
            "python",
            "adapter.py",
            "--input-manifest",
            "{input_manifest}",
            "--output-manifest",
            "{output_manifest}",
        ],
        "cwd": None,
        "network_policy": "NO_DOWNLOADS",
        "notes": "Synthetic unit-test wiring only.",
    }
    _sync_test_admission_record(modified, method)
    receipt_path, admission_files = _wire_test_admission(method, tmp_path)
    config = tmp_path / "contract.json"
    _write_json(config, modified)
    _write_test_admission_receipt(receipt_path, config, admission_files)

    input_manifest = tmp_path / "input.json"
    output_manifest = tmp_path / "output.json"
    learning_split = tmp_path / "learning-split.json"
    _write_json(
        learning_split,
        {
            "schema_version": "PETCT-LEARNING-SPLIT-v1.0",
            "status": "FROZEN_BEFORE_MODEL_SELECTION",
            "patient_count": 1,
            "case_count": 1,
            "case_counts": {"train": 0, "val": 1, "test": 0},
            "patients": [
                {
                    "patient_id": "patient-001",
                    "partition": "val",
                    "case_ids": ["case-001"],
                }
            ],
        },
    )
    input_payload = {
        "schema_version": modified["input_manifest_contract"]["schema_version"],
        "records": [
            {
                "case_id": "case-001",
                "patient_id": "patient-001",
                "split": "validation",
                "fold": 0,
                "step": 1,
                "pet_path": "pet.nii.gz",
                "ct_path": "ct.nii.gz",
                "m0_path": "m0.nii.gz",
                "fg_scribble_path": "fg.nii.gz",
                "bg_scribble_path": None,
                "original_grid_reference": "ct.nii.gz",
                "scribble_strategy": "centerline",
                    "scribble_polarity": "foreground",
                    "patient_split_receipt": {
                        "internal_partition": "val",
                        "learning_split_sha256": _sha256(learning_split),
                    },
            }
        ],
    }
    output_payload = {
        "schema_version": modified["output_manifest_contract"]["schema_version"],
        "records": [
            {
                "case_id": "case-001",
                "patient_id": "patient-001",
                "method_id": "scribbleprompt",
                "prediction_path": "prediction.nii.gz",
                "original_grid_reference": "ct.nii.gz",
                "prediction_semantics": "full_mask",
                "runtime_seconds": 1.5,
                "peak_gpu_memory_mib": None,
                "source_checkpoint_id": "unit-test-checkpoint",
                "status": "complete",
            }
        ],
    }
    _write_json(input_manifest, input_payload)

    captured: dict[str, object] = {}

    def fake_run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        captured["argv"] = argv
        captured.update(kwargs)
        _write_json(output_manifest, output_payload)
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    rc = main(
        [
            "--config",
            str(config),
            "--method",
            "scribbleprompt",
            "--input-manifest",
            str(input_manifest),
            "--output-manifest",
            str(output_manifest),
            "--learning-split",
            str(learning_split),
            "--partition",
            "val",
            "--execute",
            "--confirm",
            modified["execution_policy"]["confirmation_token"],
        ]
    )

    assert rc == 0
    assert isinstance(captured["argv"], list)
    assert captured["shell"] is False
    env = captured["env"]
    assert isinstance(env, dict)
    assert env["HF_HUB_OFFLINE"] == "1"
    assert env["TRANSFORMERS_OFFLINE"] == "1"
    assert env["PIP_NO_INDEX"] == "1"


def test_ready_execution_requires_nonempty_safe_argv() -> None:
    contract = load_and_validate_contract(CONFIG)
    modified = copy.deepcopy(contract)
    method = modified["methods"][0]
    method["execution"]["state"] = "ARGV_WIRED"
    method["execution"]["argv"] = []
    with pytest.raises(ContractError, match="non-empty argv"):
        validate_contract(modified)

    method["execution"]["argv"] = ["bash", "-c", "echo unsafe"]
    with pytest.raises(ContractError, match="shell or downloader"):
        validate_contract(modified)


def test_ready_execution_requires_runtime_admission_contract() -> None:
    contract = load_and_validate_contract(CONFIG)
    modified = copy.deepcopy(contract)
    method = next(item for item in modified["methods"] if item["id"] == "nninteractive")
    method["execution"].pop("admission")

    with pytest.raises(ContractError, match="execution.admission"):
        validate_contract(modified)


def test_runtime_admission_rejects_stale_config_and_bound_file(
    tmp_path: Path,
) -> None:
    contract = load_and_validate_contract(CONFIG)
    modified = copy.deepcopy(contract)
    method = next(item for item in modified["methods"] if item["id"] == "nninteractive")
    receipt_path, admission_files = _wire_test_admission(method, tmp_path)
    config = tmp_path / "contract.json"
    _write_json(config, modified)
    _write_test_admission_receipt(receipt_path, config, admission_files)

    checked = validate_execution_admission(method, config, variables={})
    assert checked["status"] == "PASS"
    assert checked["config"]["sha256"] == _sha256(config)

    modified["execution_policy"]["result_truth"] += " changed"
    _write_json(config, modified)
    with pytest.raises(ContractError, match="stale for the current comparator config"):
        validate_execution_admission(method, config, variables={})

    _write_test_admission_receipt(receipt_path, config, admission_files)
    admission_files["adapter_sha256"].write_text("tampered", encoding="utf-8")
    with pytest.raises(ContractError, match="hash mismatch for adapter_sha256"):
        validate_execution_admission(method, config, variables={})


def test_manifest_validation_rejects_missing_required_field() -> None:
    contract = load_and_validate_contract(CONFIG)
    document = {
        "schema_version": contract["input_manifest_contract"]["schema_version"],
        "records": [{"case_id": "case-001"}],
    }
    with pytest.raises(ContractError, match="patient_id"):
        validate_manifest(document, contract["input_manifest_contract"], "input")


def test_build_plan_refuses_unresolved_argv_placeholder(tmp_path: Path) -> None:
    contract = load_and_validate_contract(CONFIG)
    modified = copy.deepcopy(contract)
    method = next(item for item in modified["methods"] if item["id"] == "scribbleprompt")
    method["pretraining"]["checkpoint_license"] = "UNIT_TEST_LICENSE_EVIDENCE"
    method["execution"] = {
        "state": "ARGV_WIRED",
        "argv": ["python", "{missing_adapter}"],
        "cwd": None,
        "network_policy": "NO_DOWNLOADS",
        "notes": "Unit-test placeholder.",
    }
    _sync_test_admission_record(modified, method)
    _wire_test_admission(method, tmp_path)
    validate_contract(modified)
    with pytest.raises(ContractError, match="unresolved placeholder"):
        build_execution_plan(
            modified,
            "scribbleprompt",
            Path("input.json"),
            Path("output.json"),
            variables={},
            execute=True,
        )
