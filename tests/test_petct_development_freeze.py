from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import pytest
import torch


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "scripts"))
sys.path.insert(0, str(PROJECT / "scripts" / "comparators"))

from common.petct_development_freeze import (  # noqa: E402
    CHECKPOINT_BINDINGS_SCHEMA,
    EDITOR_CHECKPOINT_SCHEMA,
    INPUT_SCHEMA,
    P2T_CHECKPOINT_SCHEMA,
    REQUIRED_SINGLETON_ROLES,
    ROLE_CONTRACTS,
    TRAINED_CHECKPOINT_STATUS,
    DevelopmentFreezeError,
    _canonical_sha256,
    build_nninteractive_external_admission,
    build_final_development_freeze,
    export_frozen_checkpoint_bindings,
    resolve_frozen_artifact_path,
    resolve_frozen_checkpoint_field,
    resolve_test_external_binding,
    validate_final_development_freeze,
    validate_nninteractive_external_admission,
)
from finalize_petct_external_comparators import (  # noqa: E402
    REQUIRED_METRICS,
    ExternalCompleteError,
    build_external_complete,
    validate_external_complete,
)
from common.petct_test_access import (  # noqa: E402
    FINAL_FREEZE_CONFIRMATION,
    consume_final_freeze_grant,
    create_final_freeze_grant,
)
from common.petct_route_a_core import patient_cluster_summary  # noqa: E402


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _record(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": _sha(path)}


def _executable(path: Path, content: bytes = b"fixture-python") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    path.chmod(0o755)
    return path


def _config_and_split(
    root: Path,
    *,
    status: str = "SIX_CLASS_CANONICAL_FROZEN_CONFIRMATORY_ACTIVE_NOT_EXECUTED",
    test_policy: bool = False,
) -> tuple[Path, Path]:
    config = root / "experiment.json"
    split = root / "learning-split.json"
    learning_split: dict[str, Any] = {
        "schema_version": "PETCT-LEARNING-SPLIT-v1.0"
    }
    statistics: dict[str, Any] = {
        "confirmatory_execution_gate": "ACTIVE_AFTER_ERROR_ATLAS_FEASIBILITY_AND_EFFECT_FREEZE",
        "required_frozen_contrast_fields": [
            "family", "treatment", "comparator", "metric", "threshold_ref", "null_margin", "alternative"
        ],
        "effect_thresholds": {"p2t_min": 0.01, "editor_min": 0.01},
        "confirmatory_contrasts": [
            {"family": "editor", "treatment": "scribble_plus_intent", "comparator": "scribble_plus_operation", "metric": "authorized_target_recall", "threshold_ref": "editor_min", "null_margin": 0.0, "alternative": "greater"}
        ],
    }
    if test_policy:
        learning_split["test_access"] = (
            "exactly-once-after-all-development-freezes"
        )
        statistics["test_access"] = "exactly-once-after-freeze"
    config.write_text(
        json.dumps(
            {
                "status": status,
                "dataset": {"learning_split": learning_split},
                "statistics": statistics,
                "p2t": {
                    "primary_architecture_id": "primary_arch",
                    "simple_first_input_arms": ["full"],
                    "confirmatory_execution_gate": "ACTIVE_AFTER_ERROR_ATLAS_FEASIBILITY_AND_EFFECT_FREEZE",
                    "confirmatory_contrast": {"family": "p2t", "treatment": "full", "comparator": "full", "metric": "six_class_patient_balanced_joint_macro_f1", "threshold_ref": "p2t_min", "null_margin": 0.0, "alternative": "greater"},
                    "training": {
                        "seeds": [3407],
                        "checkpoint_criterion": "p2t-criterion",
                    },
                },
                "editor": {
                    "conditions": ["scribble_plus_intent"],
                    "training_conditions": ["scribble_plus_intent"],
                    "training": {
                        "seeds": [3407],
                        "checkpoint_criterion": "editor-criterion",
                    },
                    "primary_architecture_id": "simple_operation_conditioned_residual_unet_v2",
                },
            }
        ),
        encoding="utf-8",
    )
    split.write_text(
        json.dumps(
            {
                "schema_version": "PETCT-LEARNING-SPLIT-v1.0",
                "status": "FROZEN_BEFORE_MODEL_SELECTION",
                "patient_count": 2,
                "case_count": 2,
                "case_counts": {"train": 0, "val": 1, "test": 1},
                "patients": [
                    {
                        "patient_id": "patient-1",
                        "partition": "val",
                        "case_ids": ["case-1"],
                    },
                    {
                        "patient_id": "patient-test",
                        "partition": "test",
                        "case_ids": ["case-test"],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    return config, split


def _checkpoint(
    path: Path,
    *,
    config: Path,
    split: Path,
    training_manifest: Path,
    kind: str,
    architecture: str | None = None,
    arm: str = "full",
    condition: str | None = None,
    editor_architecture: str = "simple_operation_conditioned_residual_unet_v2",
) -> None:
    config_document = json.loads(config.read_text(encoding="utf-8"))
    common = {
        "status": TRAINED_CHECKPOINT_STATUS,
        "seed": 3407,
        "seed_registry": [3407],
        "manifest": str(training_manifest.resolve()),
        "manifest_sha256": _sha(training_manifest),
        "training_manifest": str(training_manifest.resolve()),
        "training_manifest_sha256": _sha(training_manifest),
        "learning_split": str(split.resolve()),
        "learning_split_sha256": _sha(split),
        "experiment_config": str(config.resolve()),
        "experiment_config_sha256": _sha(config),
        "input_ablation": arm,
        "checkpoint_criterion": config_document[
            "p2t" if kind == "p2t" else "editor"
        ]["training"]["checkpoint_criterion"],
        "state_dict": {"dummy": torch.tensor([1.0])},
    }
    if kind == "p2t":
        payload = {
            **common,
            "schema_version": P2T_CHECKPOINT_SCHEMA,
            "architecture_id": architecture,
            "arm_role": (
                "secondary_architecture"
                if architecture == "secondary_arch"
                else "primary"
            ),
        }
    else:
        payload = {
            **common,
            "schema_version": EDITOR_CHECKPOINT_SCHEMA,
            "condition": condition,
            "architecture_id": editor_architecture,
        }
    torch.save(payload, path)


def _artifact_manifest(
    root: Path,
    config: Path,
    split: Path,
    *,
    omit: str | None = None,
) -> tuple[Path, dict[str, Path]]:
    controlled = root / "controlled.jsonl"
    natural = root / "natural.jsonl"
    controlled.write_text("{}\n", encoding="utf-8")
    natural.write_text("{}\n", encoding="utf-8")
    p2t = root / "p2t-primary.pth"
    editor = root / "editor.pth"
    _checkpoint(
        p2t,
        config=config,
        split=split,
        training_manifest=controlled,
        kind="p2t",
        architecture="primary_arch",
    )
    _checkpoint(
        editor,
        config=config,
        split=split,
        training_manifest=natural,
        kind="editor",
        condition="scribble_plus_intent",
    )

    p2t_receipt = root / "p2t-validation.json"
    p2t_receipt.write_text(
        json.dumps(
            {
                **ROLE_CONTRACTS["p2t_validation_receipt"],
                "common": {
                    "evaluation_partition": "val",
                    "experiment_config_sha256": _sha(config),
                    "learning_split_sha256": _sha(split),
                },
                "artifact_bindings": {
                    "p2t_checkpoints": [_record(p2t)],
                    "controlled_tensor_manifest": _record(controlled),
                },
            }
        ),
        encoding="utf-8",
    )
    editor_receipt = root / "editor-validation.json"
    editor_receipt.write_text(
        json.dumps(
            {
                **ROLE_CONTRACTS["editor_validation_receipt"],
                "common": {
                    "evaluation_partition": "val",
                    "experiment_config_sha256": _sha(config),
                    "learning_split_sha256": _sha(split),
                },
                "artifact_bindings": {
                    "editor_checkpoints": [_record(editor)],
                    "natural_tensor_manifest": _record(natural),
                },
            }
        ),
        encoding="utf-8",
    )
    special = {
        "p2t_validation_receipt": p2t_receipt,
        "editor_validation_receipt": editor_receipt,
    }
    files: dict[str, Path] = {
        **special,
        "p2t_primary": p2t,
        "editor": editor,
        "controlled": controlled,
        "natural": natural,
    }
    core_prefix = root / "envs" / "petct_nnunet_v281"
    core_python = _executable(core_prefix / "bin" / "python")
    official_metrics = root / "external_runners" / "autopetv_protocol" / "metrics.py"
    official_metrics.parent.mkdir(parents=True, exist_ok=True)
    official_metrics.write_text("class MetricEvaluator: pass\n", encoding="utf-8")
    core_receipt = root / "core-environment-receipt.json"
    core_receipt.write_text(
        json.dumps(
            {
                "schema_version": "PETCT-NNUNET-ENV-v1.1",
                "status": "PREFLIGHT_PASS_PENDING_ATOMIC_EVIDENCE_PUBLICATION",
                "conda_prefix": str(core_prefix.resolve()),
                "official_autopetv_preflight": {
                    "metrics": {
                        "path": str(official_metrics.resolve()),
                        "sha256": _sha(official_metrics),
                        "required_callable": "MetricEvaluator",
                        "import_status": "PASS",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    files["core_python"] = core_python
    files["official_metrics"] = official_metrics
    files["core_environment_receipt"] = core_receipt
    records = []
    for role in sorted(REQUIRED_SINGLETON_ROLES):
        if role == omit:
            continue
        path = special.get(role, root / (role + ".json"))
        contract = ROLE_CONTRACTS[role]
        if role == "environment_receipt":
            path.write_text(
                json.dumps(
                    {
                        **contract,
                        "role": role,
                        "receipt_path": str(core_receipt.resolve()),
                        "receipt_sha256": _sha(core_receipt),
                    }
                ),
                encoding="utf-8",
            )
        elif role not in special:
            path.write_text(json.dumps({**contract, "role": role}), encoding="utf-8")
        files[role] = path
        entry: dict[str, Any] = {
            "role": role,
            "path": str(path),
            "expected_sha256": _sha(path),
            "expected_schema_version": contract["schema_version"],
            "expected_status": contract["status"],
        }
        if "target" in contract:
            entry["expected_target"] = contract["target"]
        records.append(entry)
    manifest = root / "required-artifacts.json"
    manifest.write_text(
        json.dumps({"schema_version": INPUT_SCHEMA, "artifacts": records}),
        encoding="utf-8",
    )
    return manifest, files


def _build_fixture(root: Path, *, test_policy: bool = False):
    config, split = _config_and_split(root, test_policy=test_policy)
    manifest, files = _artifact_manifest(root, config, split)
    freeze = root / "FINAL_DEVELOPMENT_FREEZE.json"
    payload = build_final_development_freeze(
        experiment_config=config,
        learning_split=split,
        required_artifacts_manifest=manifest,
        output=freeze,
    )
    return config, split, manifest, files, freeze, payload


def _source_bundle_hash(root: Path) -> tuple[str, int]:
    lines = []
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        if not path.is_file() or path.is_symlink():
            continue
        relative = path.relative_to(root)
        if any(
            part == "__pycache__" or part.endswith(".egg-info")
            for part in relative.parts
        ) or path.suffix in {".pyc", ".pyo"}:
            continue
        lines.append(f"{_sha(path)}  {relative.as_posix()}")
    encoded = ("\n".join(lines) + "\n").encode("utf-8")
    return hashlib.sha256(encoded).hexdigest(), len(lines)


def _external_admission_fixture(
    root: Path, *, experiment_config: Path, learning_split: Path
) -> dict[str, Path]:
    source = root / "external_runners" / "nninteractive" / "source"
    source.mkdir(parents=True)
    source_license = source / "LICENSE"
    source_license.write_text("Apache-2.0 fixture", encoding="utf-8")
    (source / "inference.py").write_text("VALUE = 1\n", encoding="utf-8")
    source_bundle_sha, source_count = _source_bundle_hash(source)

    model = root / "models" / "nnInteractive" / "nnInteractive_v1.0"
    (model / "fold_0").mkdir(parents=True)
    checkpoint = model / "fold_0" / "checkpoint_final.pth"
    checkpoint.write_bytes(b"fixture-checkpoint")
    model_license = model / "LICENSE"
    model_license.write_text("CC BY-NC-SA 4.0 fixture", encoding="utf-8")
    metadata: dict[str, str] = {}
    for name in ("dataset.json", "plans.json", "inference_session_class.json"):
        path = model / name
        path.write_text(json.dumps({"name": name}), encoding="utf-8")
        metadata[name] = _sha(path)

    adapter = root / "scripts" / "comparators" / "nninteractive_petct_adapter.py"
    adapter.parent.mkdir(parents=True)
    adapter.write_text("# fixture adapter\n", encoding="utf-8")
    environment = root / "envs" / "nninteractive_v1.freeze.txt"
    environment.parent.mkdir(parents=True, exist_ok=True)
    environment.write_text("fixture==1\n", encoding="utf-8")
    nninteractive_python = _executable(
        root / "envs" / "nninteractive_v1" / "bin" / "python",
        b"nninteractive-python",
    )
    core_python = _executable(
        root / "envs" / "petct_nnunet_v281" / "bin" / "python",
        b"core-python",
    )
    official_metrics = root / "external_runners" / "autopetv_protocol" / "metrics.py"
    official_metrics.parent.mkdir(parents=True, exist_ok=True)
    official_metrics.write_text("class MetricEvaluator: pass\n", encoding="utf-8")
    runtime = root / "envs" / "nninteractive_v1.READY.json"
    comparator_config = root / "configs" / "petct_external_comparators.json"
    comparator_config.parent.mkdir(parents=True)
    source_commit = "a" * 40
    comparator = {
        "schema_version": "PETCT-EXTERNAL-COMPARATOR-CONTRACT-v1.0",
        "status": "CONTRACT_ONLY_NOT_EXECUTED",
        "methods": [
            {
                "id": "nninteractive",
                "role": "RUN",
                "selection": "RUN_SECONDARY_EXPOSED_PRETRAINING",
                "spatial_dimensionality": "3D",
                "headline": {"eligible": False},
                "source": {
                    "repository": "https://example.invalid/nninteractive",
                    "pinned_commit": source_commit,
                    "license": "Apache-2.0",
                    "license_sha256": _sha(source_license),
                },
                "pretraining": {
                    "current_psma_v3_exposure": "KNOWN_PUBLIC_COHORT_EXPOSURE",
                    "local_checkpoint_availability": {
                        "status": "PRESENT_HASHED",
                        "path": str(checkpoint.relative_to(root)),
                        "sha256": _sha(checkpoint),
                        "license": "CC BY-NC-SA 4.0",
                        "license_file": str(model_license.relative_to(root)),
                        "license_sha256": _sha(model_license),
                    },
                },
                "execution": {
                    "state": "ARGV_WIRED",
                    "network_policy": "NO_DOWNLOADS",
                    "argv": [
                        "{project_root}/envs/nninteractive_v1/bin/python",
                        "{project_root}/scripts/comparators/nninteractive_petct_adapter.py",
                    ],
                    "admission": {
                        "receipt": "{project_root}/envs/nninteractive_v1.READY.json",
                        "schema_version": "PETCT-NNINTERACTIVE-ENV-v1.1",
                        "status": "PASS",
                        "config_sha256_field": "config_sha256",
                        "required_pass_fields": [
                            "model_load_smoke",
                            "initial_m0_api_smoke",
                            "scribble_api_smoke",
                            "adapter_cli_smoke",
                        ],
                        "exact_fields": {
                            "source_commit": source_commit,
                            "source_bundle_sha256": source_bundle_sha,
                            "source_license": "Apache-2.0",
                            "license": "CC BY-NC-SA 4.0",
                            "synthetic_only": True,
                            "scientific_prediction_produced": False,
                            "network_policy_at_runtime": "NO_DOWNLOADS",
                        },
                        "file_sha256_fields": {
                            "adapter_sha256": "{project_root}/scripts/comparators/nninteractive_petct_adapter.py",
                            "checkpoint_sha256": "{project_root}/models/nnInteractive/nnInteractive_v1.0/fold_0/checkpoint_final.pth",
                            "license_sha256": "{project_root}/models/nnInteractive/nnInteractive_v1.0/LICENSE",
                            "environment_freeze_sha256": "{project_root}/envs/nninteractive_v1.freeze.txt",
                            "source_license_sha256": "{project_root}/external_runners/nninteractive/source/LICENSE",
                        },
                    },
                },
            }
        ],
    }
    comparator_config.write_text(json.dumps(comparator), encoding="utf-8")
    runtime.write_text(
        json.dumps(
            {
                "schema_version": "PETCT-NNINTERACTIVE-ENV-v1.1",
                "status": "PASS",
                "cuda_available": True,
                "smoke_device": "cuda:0",
                "conda_prefix": str(nninteractive_python.parent.parent.resolve()),
                "config_sha256": _sha(comparator_config),
                "model_load_smoke": "PASS",
                "initial_m0_api_smoke": "PASS",
                "scribble_api_smoke": "PASS",
                "adapter_cli_smoke": "PASS",
                "source_commit": source_commit,
                "source_bundle_sha256": source_bundle_sha,
                "source_bundle_file_count": source_count,
                "source_license": "Apache-2.0",
                "license": "CC BY-NC-SA 4.0",
                "synthetic_only": True,
                "scientific_prediction_produced": False,
                "network_policy_at_runtime": "NO_DOWNLOADS",
                "adapter_sha256": _sha(adapter),
                "checkpoint_sha256": _sha(checkpoint),
                "license_sha256": _sha(model_license),
                "environment_freeze_sha256": _sha(environment),
                "source_license_sha256": _sha(source_license),
                "model_metadata_sha256": metadata,
            }
        ),
        encoding="utf-8",
    )

    run_root = root / "external-val"
    artifacts = run_root / "artifacts"
    metrics = run_root / "metrics"
    artifacts.mkdir(parents=True)
    metrics.mkdir()
    natural_episode_manifest = root / "natural_episodes.jsonl"
    natural_episode_manifest.write_text(
        json.dumps(
            {
                "episode_id": "episode-1",
                "case_id": "case-1",
                "patient_id": "patient-1",
                "partition": "val",
                "learning_split_sha256": _sha(learning_split),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    input_manifest = artifacts / "external_comparator_input.json"
    input_manifest.write_text(
        json.dumps(
            {
                "schema_version": "PETCT-EXTERNAL-COMPARATOR-INPUT-v1.0",
                "status": "FROZEN_INPUT_READY",
                "partition": "validation",
                "record_count": 1,
                "patient_count": 1,
                "provenance": {
                    "learning_split_sha256": _sha(learning_split),
                    "natural_episode_manifest_sha256": _sha(
                        natural_episode_manifest
                    ),
                },
                "records": [
                    {
                        "case_id": "case-1",
                        "patient_id": "patient-1",
                        "split": "validation",
                        "episode_id": "episode-1",
                        "patient_split_receipt": {
                            "internal_partition": "val",
                            "learning_split_sha256": _sha(learning_split),
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    checkpoint_sha = _sha(checkpoint)
    for policy, role in (
        ("union_with_m0", "POSITIVE_ONLY_DIAGNOSTIC"),
        ("native_full_mask", "NATIVE_DIAGNOSTIC"),
    ):
        prediction_dir = run_root / "predictions" / f"nninteractive_{policy}"
        prediction_dir.mkdir(parents=True)
        prediction = prediction_dir / "case-1.nii.gz"
        prediction.write_bytes(f"prediction-{policy}".encode("utf-8"))
        output_manifest = artifacts / f"nninteractive_{policy}_output.json"
        output_manifest.write_text(
            json.dumps(
                {
                    "schema_version": "PETCT-EXTERNAL-COMPARATOR-OUTPUT-v1.0",
                    "method_id": "nninteractive",
                    "output_policy": policy,
                    "learning_split_sha256": _sha(learning_split),
                    "test_access_receipt_sha256": None,
                    "records": [
                        {
                            "case_id": "case-1",
                            "patient_id": "patient-1",
                            "method_id": "nninteractive",
                            "output_policy": policy,
                            "status": "complete",
                            "prediction_path": str(prediction.resolve()),
                            "prediction_sha256": _sha(prediction),
                            "checkpoint_sha256": checkpoint_sha,
                            "source_checkpoint_id": f"nnInteractive_v1.0:{checkpoint_sha}",
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        rows = metrics / f"nninteractive_{policy}_rows.jsonl"
        metric_row = {
            "case_id": "case-1",
            "episode_id": "episode-1",
            "patient_id": "patient-1",
            "split": "validation",
            "method_id": "nninteractive",
            "output_policy": policy,
            "status": "complete",
            **{name: 0.0 for name in REQUIRED_METRICS},
        }
        rows.write_text(
            json.dumps(metric_row) + "\n",
            encoding="utf-8",
        )
        summary = metrics / f"nninteractive_{policy}_summary.json"
        summary.write_text(
            json.dumps(
                {
                    "schema_version": "PETCT-EXTERNAL-COMPARATOR-METRICS-v1.0",
                    "status": "COMPLETE",
                    "method_id": "nninteractive",
                    "output_policy": policy,
                    "spatial_dimensionality": "3D",
                    "comparison_role": role,
                    "cross_dimensional_pooling": "FORBIDDEN",
                    "separate_fairness_table": True,
                    "input_manifest_sha256": _sha(input_manifest),
                    "output_manifest_sha256": _sha(output_manifest),
                    "experiment_config_sha256": _sha(experiment_config),
                    "learning_split_sha256": _sha(learning_split),
                    "record_count": 1,
                    "patient_count": 1,
                    "complete_count": 1,
                    "failed_count": 0,
                    "pretraining_exposure": "KNOWN_PUBLIC_COHORT_EXPOSURE",
                    "headline_eligible": False,
                    "natural_episode_manifest_sha256": _sha(
                        natural_episode_manifest
                    ),
                    "test_access_receipt_sha256": None,
                    "official_metrics_sha256": _sha(official_metrics),
                    "metric_rows_sha256": _sha(rows),
                    "patient_clustered": {
                        name: patient_cluster_summary([metric_row], name)
                        for name in REQUIRED_METRICS
                    },
                }
            ),
            encoding="utf-8",
        )
    complete_path = artifacts / "EXTERNAL_COMPARATORS_COMPLETE.json"
    build_external_complete(
        run_root=run_root,
        partition="val",
        selected_methods=["nninteractive"],
        comparator_config=comparator_config,
        experiment_config=experiment_config,
        learning_split=learning_split,
        input_manifest=input_manifest,
        natural_episode_manifest=natural_episode_manifest,
        runtime_receipts={"nninteractive": runtime},
        core_python=core_python,
        official_metrics=official_metrics,
        nninteractive_python=nninteractive_python,
        output=complete_path,
    )
    admission = artifacts / "NNINTERACTIVE_EXTERNAL_ADMISSION.json"
    build_nninteractive_external_admission(
        comparator_config=comparator_config,
        experiment_config=experiment_config,
        learning_split=learning_split,
        validation_complete=complete_path,
        output=admission,
    )
    return {
        "admission": admission,
        "adapter": adapter,
        "checkpoint": checkpoint,
        "comparator_config": comparator_config,
        "complete": complete_path,
        "environment": environment,
        "core_python": core_python,
        "official_metrics": official_metrics,
        "nninteractive_python": nninteractive_python,
        "natural_episode_manifest": natural_episode_manifest,
        "runtime": runtime,
        "source_file": source / "inference.py",
    }


def _consume_fixture_grant(
    root: Path, *, config: Path, split: Path, freeze: Path
) -> tuple[Path, Path, Path]:
    grant = root / "TEST_ACCESS_GRANT.json"
    create_final_freeze_grant(
        experiment_config=config,
        learning_split=split,
        final_development_freeze=freeze,
        grant_path=grant,
        authorized_by="director",
        confirmation=FINAL_FREEZE_CONFIRMATION,
    )
    run_root = root / "formal-test"
    run_root.mkdir()
    receipt = run_root / "TEST_ACCESS_CONSUMED.json"
    ledger = root / "ledger"
    consume_final_freeze_grant(
        grant_path=grant,
        run_root=run_root,
        receipt_path=receipt,
        ledger_root=ledger,
    )
    return receipt, run_root, ledger


def _refresh_manifest_hash(manifest: Path, role: str, path: Path) -> None:
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    for entry in payload["artifacts"]:
        if entry["role"] == role:
            entry["expected_sha256"] = _sha(path)
    manifest.write_text(json.dumps(payload), encoding="utf-8")


def test_draft_config_cannot_be_misrepresented_as_final_freeze(tmp_path: Path) -> None:
    config, split = _config_and_split(tmp_path, status="DRAFT")
    manifest, _ = _artifact_manifest(tmp_path, config, split)
    with pytest.raises(DevelopmentFreezeError, match="six-class"):
        build_final_development_freeze(
            experiment_config=config,
            learning_split=split,
            required_artifacts_manifest=manifest,
            output=tmp_path / "FINAL_DEVELOPMENT_FREEZE.json",
        )


def test_current_confirmatory_blocked_v2_config_cannot_publish_final_freeze(
    tmp_path: Path,
) -> None:
    config, split = _config_and_split(tmp_path)
    payload = json.loads(config.read_text(encoding="utf-8"))
    payload["status"] = (
        "SIX_CLASS_CANONICAL_FROZEN_LOCAL_IMPLEMENTATION_PRESENT_"
        "CONFIRMATORY_BLOCKED_NOT_EXECUTED"
    )
    payload["p2t"]["confirmatory_execution_gate"] = (
        "BLOCKED_UNTIL_ERROR_ATLAS_FEASIBILITY_AND_EFFECT_FREEZE"
    )
    payload["statistics"]["confirmatory_execution_gate"] = (
        "BLOCKED_UNTIL_ERROR_ATLAS_FEASIBILITY_AND_EFFECT_FREEZE"
    )
    config.write_text(json.dumps(payload), encoding="utf-8")
    manifest, _ = _artifact_manifest(tmp_path, config, split)

    with pytest.raises(DevelopmentFreezeError, match="error-atlas feasibility"):
        build_final_development_freeze(
            experiment_config=config,
            learning_split=split,
            required_artifacts_manifest=manifest,
            output=tmp_path / "FINAL_DEVELOPMENT_FREEZE.json",
        )


def test_missing_required_artifact_is_rejected(tmp_path: Path) -> None:
    config, split = _config_and_split(tmp_path)
    manifest, _ = _artifact_manifest(tmp_path, config, split, omit="statistics_plan")
    with pytest.raises(DevelopmentFreezeError, match="statistics_plan"):
        build_final_development_freeze(
            experiment_config=config,
            learning_split=split,
            required_artifacts_manifest=manifest,
            output=tmp_path / "FINAL_DEVELOPMENT_FREEZE.json",
        )


def test_arbitrary_pass_json_cannot_impersonate_oof_receipt(tmp_path: Path) -> None:
    config, split = _config_and_split(tmp_path)
    manifest, files = _artifact_manifest(tmp_path, config, split)
    fake = files["m0_oof_receipt"]
    fake.write_text(
        json.dumps({"schema_version": "PETCT-M0-OOF-READY-v1.0", "status": "PASS"}),
        encoding="utf-8",
    )
    _refresh_manifest_hash(manifest, "m0_oof_receipt", fake)
    with pytest.raises(DevelopmentFreezeError, match="invalid status"):
        build_final_development_freeze(
            experiment_config=config,
            learning_split=split,
            required_artifacts_manifest=manifest,
            output=tmp_path / "FINAL_DEVELOPMENT_FREEZE.json",
        )


def test_arbitrary_file_and_self_declared_role_cannot_satisfy_freeze(tmp_path: Path) -> None:
    config, split = _config_and_split(tmp_path)
    manifest, _ = _artifact_manifest(tmp_path, config, split)
    arbitrary = tmp_path / "arbitrary.pth"
    arbitrary.write_bytes(b"anything")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["selected_checkpoint_roles"] = ["selected_checkpoint:p2t:fake:full:seed1"]
    payload["artifacts"].append(
        {"role": payload["selected_checkpoint_roles"][0], "path": str(arbitrary)}
    )
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(DevelopmentFreezeError, match="derived from validation receipts"):
        build_final_development_freeze(
            experiment_config=config,
            learning_split=split,
            required_artifacts_manifest=manifest,
            output=tmp_path / "FINAL_DEVELOPMENT_FREEZE.json",
        )


def test_missing_or_swapped_checkpoint_role_is_rejected(tmp_path: Path) -> None:
    config, split = _config_and_split(tmp_path)
    manifest, files = _artifact_manifest(tmp_path, config, split)
    receipt = files["p2t_validation_receipt"]
    document = json.loads(receipt.read_text(encoding="utf-8"))
    document["artifact_bindings"]["p2t_checkpoints"] = []
    receipt.write_text(json.dumps(document), encoding="utf-8")
    _refresh_manifest_hash(manifest, "p2t_validation_receipt", receipt)
    with pytest.raises(DevelopmentFreezeError, match="grid is not exact"):
        build_final_development_freeze(
            experiment_config=config,
            learning_split=split,
            required_artifacts_manifest=manifest,
            output=tmp_path / "FINAL_DEVELOPMENT_FREEZE.json",
        )


def test_extra_checkpoint_descriptor_is_rejected(tmp_path: Path) -> None:
    config, split = _config_and_split(tmp_path)
    manifest, files = _artifact_manifest(tmp_path, config, split)
    extra = tmp_path / "extra.pth"
    _checkpoint(
        extra,
        config=config,
        split=split,
        training_manifest=files["controlled"],
        kind="p2t",
        architecture="primary_arch",
        arm="not_registered",
    )
    receipt = files["p2t_validation_receipt"]
    document = json.loads(receipt.read_text(encoding="utf-8"))
    document["artifact_bindings"]["p2t_checkpoints"].append(_record(extra))
    receipt.write_text(json.dumps(document), encoding="utf-8")
    _refresh_manifest_hash(manifest, "p2t_validation_receipt", receipt)
    with pytest.raises(DevelopmentFreezeError, match="extra or swapped"):
        build_final_development_freeze(
            experiment_config=config,
            learning_split=split,
            required_artifacts_manifest=manifest,
            output=tmp_path / "FINAL_DEVELOPMENT_FREEZE.json",
        )


def test_secondary_checkpoint_role_is_not_part_of_v2_freeze() -> None:
    assert "p2t_secondary_validation_receipt" not in REQUIRED_SINGLETON_ROLES
    assert "p2t_secondary_validation_receipt" not in ROLE_CONTRACTS


def test_freeze_recomputes_every_artifact_and_rejects_later_tamper(tmp_path: Path) -> None:
    config, split, _, files, freeze, built = _build_fixture(tmp_path)
    assert built["status"] == "ALL_DEVELOPMENT_FROZEN"
    assert built["selected_checkpoint_count"] == 2
    assert set(built["selected_checkpoint_roles"]) == {
        "selected_checkpoint:p2t:primary_arch:full:seed3407",
        "selected_checkpoint:editor:scribble_plus_intent:simple_operation_conditioned_residual_unet_v2:seed3407",
    }
    assert (
        validate_final_development_freeze(
            freeze, experiment_config=config, learning_split=split
        )["freeze_sha256"]
        == built["freeze_sha256"]
    )
    files["p2t_primary"].write_bytes(b"changed-after-freeze")
    with pytest.raises(DevelopmentFreezeError, match="embedded SHA-256 mismatch"):
        validate_final_development_freeze(
            freeze, experiment_config=config, learning_split=split
        )


def test_final_freeze_preserves_the_training_time_config_bytes(tmp_path: Path) -> None:
    config, split = _config_and_split(tmp_path)
    config_sha_before = _sha(config)
    manifest, _ = _artifact_manifest(tmp_path, config, split)
    built = build_final_development_freeze(
        experiment_config=config,
        learning_split=split,
        required_artifacts_manifest=manifest,
        output=tmp_path / "FINAL_DEVELOPMENT_FREEZE.json",
    )
    assert _sha(config) == config_sha_before
    assert built["status"] == "ALL_DEVELOPMENT_FROZEN"
    assert built["required_artifacts"][0]["sha256"] == config_sha_before


def test_exported_bindings_are_exact_and_post_freeze_file_cannot_be_used(tmp_path: Path) -> None:
    config, split, _, files, freeze, built = _build_fixture(tmp_path)
    output = tmp_path / "bindings.json"
    bindings = export_frozen_checkpoint_bindings(
        freeze,
        experiment_config=config,
        learning_split=split,
        output=output,
    )
    assert bindings["schema_version"] == CHECKPOINT_BINDINGS_SCHEMA
    assert bindings["checkpoint_inventory_sha256"] == built[
        "checkpoint_inventory_sha256"
    ]
    assert [row["role"] for row in bindings["checkpoints"]] == built[
        "selected_checkpoint_roles"
    ]
    post_freeze = tmp_path / "post-freeze.pth"
    post_freeze.write_bytes(b"new checkpoint")
    assert str(post_freeze.resolve()) not in {
        row["path"] for row in bindings["checkpoints"]
    }
    with pytest.raises(DevelopmentFreezeError, match="exactly one role"):
        resolve_frozen_checkpoint_field(
            output,
            role="selected_checkpoint:p2t:post_freeze:full:seed3407",
            field="path",
        )
    # Formal-test non-model prerequisites are resolved from the same freeze.
    assert resolve_frozen_artifact_path(
        output, role="m0_oof_receipt"
    ) == str(files["m0_oof_receipt"].resolve())
    assert resolve_frozen_artifact_path(
        output, role="environment_receipt"
    ) == str(files["environment_receipt"].resolve())
    forged = json.loads(output.read_text(encoding="utf-8"))
    forged["checkpoints"][0]["path"] = str(post_freeze.resolve())
    unsigned = {key: value for key, value in forged.items() if key != "bindings_sha256"}
    forged["bindings_sha256"] = _canonical_sha256(unsigned)
    output.write_text(json.dumps(forged), encoding="utf-8")
    with pytest.raises(DevelopmentFreezeError, match="differ from final freeze"):
        resolve_frozen_checkpoint_field(
            output,
            role=built["selected_checkpoint_roles"][0],
            field="path",
        )


def test_freeze_bound_environment_replacement_is_rejected(tmp_path: Path) -> None:
    config, split, _, files, freeze, _ = _build_fixture(tmp_path)
    output = tmp_path / "bindings.json"
    export_frozen_checkpoint_bindings(
        freeze,
        experiment_config=config,
        learning_split=split,
        output=output,
    )
    files["environment_receipt"].write_text("replaced", encoding="utf-8")
    with pytest.raises(DevelopmentFreezeError, match="changed after final freeze"):
        resolve_frozen_artifact_path(output, role="environment_receipt")


def test_optional_nninteractive_admission_is_bound_without_becoming_primary_gate(
    tmp_path: Path,
) -> None:
    config, split = _config_and_split(tmp_path)
    manifest, _ = _artifact_manifest(tmp_path, config, split)
    external = _external_admission_fixture(
        tmp_path, experiment_config=config, learning_split=split
    )
    freeze = tmp_path / "FINAL_DEVELOPMENT_FREEZE.json"
    payload = build_final_development_freeze(
        experiment_config=config,
        learning_split=split,
        required_artifacts_manifest=manifest,
        external_method_admission=external["admission"],
        output=freeze,
    )
    assert [row["role"] for row in payload["external_method_bindings"]] == [
        "selected_external_method:nninteractive"
    ]
    assert payload["external_method_bindings"][0]["headline_eligible"] is False
    validate_final_development_freeze(
        freeze, experiment_config=config, learning_split=split
    )

    other_root = tmp_path / "primary-only"
    other_root.mkdir()
    primary_config, primary_split = _config_and_split(other_root)
    primary_manifest, _ = _artifact_manifest(
        other_root, primary_config, primary_split
    )
    primary = build_final_development_freeze(
        experiment_config=primary_config,
        learning_split=primary_split,
        required_artifacts_manifest=primary_manifest,
        output=other_root / "FINAL_DEVELOPMENT_FREEZE.json",
    )
    assert primary["external_method_bindings"] == []


@pytest.mark.parametrize(
    "drift_key",
    [
        "adapter",
        "checkpoint",
        "runtime",
        "comparator_config",
        "environment",
        "source_file",
        "nninteractive_python",
    ],
)
def test_external_admission_rejects_post_freeze_evidence_drift(
    tmp_path: Path, drift_key: str
) -> None:
    config, split = _config_and_split(tmp_path)
    external = _external_admission_fixture(
        tmp_path, experiment_config=config, learning_split=split
    )
    external[drift_key].write_bytes(external[drift_key].read_bytes() + b"drift")
    with pytest.raises(DevelopmentFreezeError):
        validate_nninteractive_external_admission(external["admission"])


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        (("runtime_receipt", "path"), "adapter"),
        (("comparator_config", "path"), "runtime"),
        (("adapter", "path"), "checkpoint"),
        (("model", "checkpoint", "path"), "environment"),
    ],
)
def test_external_admission_rejects_swapped_bound_roles_even_when_resealed(
    tmp_path: Path, field: tuple[str, ...], replacement: str
) -> None:
    config, split = _config_and_split(tmp_path)
    external = _external_admission_fixture(
        tmp_path, experiment_config=config, learning_split=split
    )
    payload = json.loads(external["admission"].read_text(encoding="utf-8"))
    target: dict[str, Any] = payload
    for key in field[:-1]:
        target = target[key]
    target[field[-1]] = str(external[replacement].resolve())
    unsigned = {key: value for key, value in payload.items() if key != "admission_sha256"}
    payload["admission_sha256"] = _canonical_sha256(unsigned)
    external["admission"].write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(DevelopmentFreezeError):
        validate_nninteractive_external_admission(external["admission"])


def test_external_admission_rejects_extra_key_hash_bypass_and_method_relabel(
    tmp_path: Path,
) -> None:
    config, split = _config_and_split(tmp_path)
    external = _external_admission_fixture(
        tmp_path, experiment_config=config, learning_split=split
    )
    payload = json.loads(external["admission"].read_text(encoding="utf-8"))
    payload["arbitrary_selected_method"] = "scribbleprompt"
    unsigned = {key: value for key, value in payload.items() if key != "admission_sha256"}
    payload["admission_sha256"] = _canonical_sha256(unsigned)
    external["admission"].write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(DevelopmentFreezeError, match="recursively rehashed evidence"):
        validate_nninteractive_external_admission(external["admission"])

    other = tmp_path / "method-relabel"
    other.mkdir()
    other_config, other_split = _config_and_split(other)
    relabeled = _external_admission_fixture(
        other, experiment_config=other_config, learning_split=other_split
    )
    payload = json.loads(relabeled["admission"].read_text(encoding="utf-8"))
    payload["method_id"] = "scribbleprompt"
    payload["role"] = "selected_external_method:scribbleprompt"
    unsigned = {key: value for key, value in payload.items() if key != "admission_sha256"}
    payload["admission_sha256"] = _canonical_sha256(unsigned)
    relabeled["admission"].write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(DevelopmentFreezeError, match="contract mismatch"):
        validate_nninteractive_external_admission(relabeled["admission"])


def test_external_complete_rejects_extra_key_reseal_and_prediction_leaf_drift(
    tmp_path: Path,
) -> None:
    config, split = _config_and_split(tmp_path)
    external = _external_admission_fixture(
        tmp_path, experiment_config=config, learning_split=split
    )
    complete = json.loads(external["complete"].read_text(encoding="utf-8"))
    complete["invented_success"] = True
    unsigned = {key: value for key, value in complete.items() if key != "receipt_sha256"}
    complete["receipt_sha256"] = _canonical_sha256(unsigned)
    external["complete"].write_text(json.dumps(complete), encoding="utf-8")
    with pytest.raises(ExternalCompleteError, match="recursively rehashed artifacts"):
        validate_external_complete(external["complete"])

    # Restore a fresh closure, then alter one prediction leaf without touching
    # its manifest/receipt.  Recursive validation must still fail.
    other = tmp_path / "leaf-drift"
    other.mkdir()
    other_config, other_split = _config_and_split(other)
    fresh = _external_admission_fixture(
        other, experiment_config=other_config, learning_split=other_split
    )
    prediction = (
        other
        / "external-val"
        / "predictions"
        / "nninteractive_union_with_m0"
        / "case-1.nii.gz"
    )
    prediction.write_bytes(prediction.read_bytes() + b"drift")
    with pytest.raises(ExternalCompleteError):
        validate_external_complete(fresh["complete"])

    extra_root = tmp_path / "extra-leaf"
    extra_root.mkdir()
    extra_config, extra_split = _config_and_split(extra_root)
    extra = _external_admission_fixture(
        extra_root, experiment_config=extra_config, learning_split=extra_split
    )
    extra_prediction = (
        extra_root
        / "external-val"
        / "predictions"
        / "nninteractive_union_with_m0"
        / "unreferenced.nii.gz"
    )
    extra_prediction.write_bytes(b"unreferenced")
    with pytest.raises(ExternalCompleteError, match="unreferenced leaf"):
        validate_external_complete(extra["complete"])

    metric_root = tmp_path / "missing-metric"
    metric_root.mkdir()
    metric_config, metric_split = _config_and_split(metric_root)
    missing_metric = _external_admission_fixture(
        metric_root, experiment_config=metric_config, learning_split=metric_split
    )
    metric_rows = (
        metric_root
        / "external-val"
        / "metrics"
        / "nninteractive_union_with_m0_rows.jsonl"
    )
    metric_row = json.loads(metric_rows.read_text(encoding="utf-8"))
    metric_row.pop("dice")
    metric_rows.write_text(json.dumps(metric_row) + "\n", encoding="utf-8")
    with pytest.raises(ExternalCompleteError, match="metric row"):
        validate_external_complete(missing_metric["complete"])

    coverage_root = tmp_path / "omitted-case"
    coverage_root.mkdir()
    coverage_config, coverage_split = _config_and_split(coverage_root)
    omitted = _external_admission_fixture(
        coverage_root,
        experiment_config=coverage_config,
        learning_split=coverage_split,
    )
    coverage_input = (
        coverage_root / "external-val" / "artifacts" / "external_comparator_input.json"
    )
    coverage = json.loads(coverage_input.read_text(encoding="utf-8"))
    coverage["records"].append(
        {
            "case_id": "case-1",
            "patient_id": "patient-1",
            "split": "validation",
            "episode_id": "invented-episode",
            "patient_split_receipt": {
                "internal_partition": "val",
                "learning_split_sha256": _sha(coverage_split),
            },
        }
    )
    coverage["record_count"] = 2
    coverage_input.write_text(json.dumps(coverage), encoding="utf-8")
    with pytest.raises(ExternalCompleteError, match="episode-subset inventory"):
        validate_external_complete(omitted["complete"])


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("patient_id", "patient-test", "patient differs from frozen split"),
        ("split", "test", "another partition"),
    ],
)
def test_external_complete_rejects_input_patient_or_partition_relabel(
    tmp_path: Path, field: str, value: str, message: str
) -> None:
    config, split = _config_and_split(tmp_path)
    external = _external_admission_fixture(
        tmp_path, experiment_config=config, learning_split=split
    )
    input_path = (
        tmp_path / "external-val" / "artifacts" / "external_comparator_input.json"
    )
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    payload["records"][0][field] = value
    input_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ExternalCompleteError, match=message):
        validate_external_complete(external["complete"])


def test_external_complete_rejects_omitted_frozen_episode(tmp_path: Path) -> None:
    config, split = _config_and_split(tmp_path)
    external = _external_admission_fixture(
        tmp_path, experiment_config=config, learning_split=split
    )
    natural = external["natural_episode_manifest"]
    with natural.open("a", encoding="utf-8") as stream:
        stream.write(
            json.dumps(
                {
                    "episode_id": "episode-omitted",
                    "case_id": "case-1",
                    "patient_id": "patient-1",
                    "partition": "val",
                    "learning_split_sha256": _sha(split),
                }
            )
            + "\n"
        )
    input_path = (
        tmp_path / "external-val" / "artifacts" / "external_comparator_input.json"
    )
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    payload["provenance"]["natural_episode_manifest_sha256"] = _sha(natural)
    input_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ExternalCompleteError, match="episode-subset inventory"):
        validate_external_complete(external["complete"])


@pytest.mark.parametrize("policy", ["union_with_m0", "native_full_mask"])
def test_external_complete_rejects_each_policy_checkpoint_relabel(
    tmp_path: Path, policy: str
) -> None:
    config, split = _config_and_split(tmp_path)
    external = _external_admission_fixture(
        tmp_path, experiment_config=config, learning_split=split
    )
    artifacts = tmp_path / "external-val" / "artifacts"
    output_path = artifacts / f"nninteractive_{policy}_output.json"
    output = json.loads(output_path.read_text(encoding="utf-8"))
    output["records"][0]["checkpoint_sha256"] = "0" * 64
    output["records"][0]["source_checkpoint_id"] = "nnInteractive_v1.0:" + "0" * 64
    output_path.write_text(json.dumps(output), encoding="utf-8")
    summary_path = (
        tmp_path / "external-val" / "metrics" / f"nninteractive_{policy}_summary.json"
    )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["output_manifest_sha256"] = _sha(output_path)
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(ExternalCompleteError, match="current admitted checkpoint"):
        validate_external_complete(external["complete"])


@pytest.mark.parametrize(
    ("metric", "value", "message"),
    [
        ("dice", True, "finite real numeric"),
        ("dice", "0.5", "finite real numeric"),
        ("dice", None, "finite real numeric"),
        ("dice", 1.5, "valid range"),
        ("runtime_seconds", -1.0, "valid range"),
    ],
)
def test_external_complete_rejects_invalid_required_metric_values(
    tmp_path: Path, metric: str, value: Any, message: str
) -> None:
    config, split = _config_and_split(tmp_path)
    external = _external_admission_fixture(
        tmp_path, experiment_config=config, learning_split=split
    )
    rows_path = (
        tmp_path
        / "external-val"
        / "metrics"
        / "nninteractive_union_with_m0_rows.jsonl"
    )
    row = json.loads(rows_path.read_text(encoding="utf-8"))
    row[metric] = value
    rows_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    summary_path = rows_path.with_name("nninteractive_union_with_m0_summary.json")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["metric_rows_sha256"] = _sha(rows_path)
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(ExternalCompleteError, match=message):
        validate_external_complete(external["complete"])


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("patient_id", "patient-test"),
        ("split", "test"),
        ("method_id", "scribbleprompt"),
        ("output_policy", "native_full_mask"),
    ],
)
def test_external_complete_rejects_metric_row_provenance_relabel(
    tmp_path: Path, field: str, value: str
) -> None:
    config, split = _config_and_split(tmp_path)
    external = _external_admission_fixture(
        tmp_path, experiment_config=config, learning_split=split
    )
    rows_path = (
        tmp_path
        / "external-val"
        / "metrics"
        / "nninteractive_union_with_m0_rows.jsonl"
    )
    row = json.loads(rows_path.read_text(encoding="utf-8"))
    row[field] = value
    rows_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    summary_path = rows_path.with_name("nninteractive_union_with_m0_summary.json")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["metric_rows_sha256"] = _sha(rows_path)
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(ExternalCompleteError, match="metric row method/policy"):
        validate_external_complete(external["complete"])


def test_external_complete_recomputes_patient_clustered_from_rows(
    tmp_path: Path,
) -> None:
    config, split = _config_and_split(tmp_path)
    external = _external_admission_fixture(
        tmp_path, experiment_config=config, learning_split=split
    )
    summary_path = (
        tmp_path
        / "external-val"
        / "metrics"
        / "nninteractive_union_with_m0_summary.json"
    )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["patient_clustered"]["dice"]["mean"] = 0.75
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(ExternalCompleteError, match="recomputed rows"):
        validate_external_complete(external["complete"])


def test_consumed_receipt_resolves_exact_external_role_and_missing_role_fails_closed(
    tmp_path: Path,
) -> None:
    admitted_root = tmp_path / "admitted"
    admitted_root.mkdir()
    config, split = _config_and_split(admitted_root, test_policy=True)
    manifest, _ = _artifact_manifest(admitted_root, config, split)
    external = _external_admission_fixture(
        admitted_root, experiment_config=config, learning_split=split
    )
    freeze = admitted_root / "FINAL_DEVELOPMENT_FREEZE.json"
    build_final_development_freeze(
        experiment_config=config,
        learning_split=split,
        required_artifacts_manifest=manifest,
        external_method_admission=external["admission"],
        output=freeze,
    )
    receipt, run_root, ledger = _consume_fixture_grant(
        admitted_root, config=config, split=split, freeze=freeze
    )
    resolved = resolve_test_external_binding(
        test_access_receipt=receipt,
        experiment_config=config,
        run_root=run_root,
        method_id="nninteractive",
        ledger_root=ledger,
    )
    assert resolved["admission"]["path"] == str(external["admission"].resolve())
    assert resolved["model_checkpoint"]["path"] == str(
        external["checkpoint"].resolve()
    )
    assert resolved["nninteractive_python"]["path"] == str(
        external["nninteractive_python"].resolve()
    )
    assert resolved["core_python"]["path"] == str(
        external["core_python"].resolve()
    )
    assert resolved["official_metrics"]["path"] == str(
        external["official_metrics"].resolve()
    )
    assert resolved["evaluation_only"] is True
    with pytest.raises(DevelopmentFreezeError, match="only for nninteractive"):
        resolve_test_external_binding(
            test_access_receipt=receipt,
            experiment_config=config,
            run_root=run_root,
            method_id="scribbleprompt",
            ledger_root=ledger,
        )

    missing_root = tmp_path / "missing"
    missing_root.mkdir()
    missing_config, missing_split = _config_and_split(missing_root, test_policy=True)
    missing_manifest, _ = _artifact_manifest(
        missing_root, missing_config, missing_split
    )
    missing_freeze = missing_root / "FINAL_DEVELOPMENT_FREEZE.json"
    build_final_development_freeze(
        experiment_config=missing_config,
        learning_split=missing_split,
        required_artifacts_manifest=missing_manifest,
        output=missing_freeze,
    )
    missing_receipt, missing_run, missing_ledger = _consume_fixture_grant(
        missing_root,
        config=missing_config,
        split=missing_split,
        freeze=missing_freeze,
    )
    with pytest.raises(DevelopmentFreezeError, match="does not contain nninteractive"):
        resolve_test_external_binding(
            test_access_receipt=missing_receipt,
            experiment_config=missing_config,
            run_root=missing_run,
            method_id="nninteractive",
            ledger_root=missing_ledger,
        )


def test_formal_external_complete_rejects_actual_checkpoint_different_from_freeze(
    tmp_path: Path,
) -> None:
    config, split = _config_and_split(tmp_path, test_policy=True)
    manifest, _ = _artifact_manifest(tmp_path, config, split)
    external = _external_admission_fixture(
        tmp_path, experiment_config=config, learning_split=split
    )
    freeze = tmp_path / "FINAL_DEVELOPMENT_FREEZE.json"
    build_final_development_freeze(
        experiment_config=config,
        learning_split=split,
        required_artifacts_manifest=manifest,
        external_method_admission=external["admission"],
        output=freeze,
    )
    test_receipt, route_root, ledger = _consume_fixture_grant(
        tmp_path, config=config, split=split, freeze=freeze
    )
    preflight_root = route_root / "preflight-external"
    missing_input = preflight_root / "artifacts" / "missing-input.json"
    invalid_receipt = route_root / "INVALID_TEST_ACCESS.json"
    invalid_receipt.write_text("{}", encoding="utf-8")
    with pytest.raises(ExternalCompleteError, match="access/freeze binding is invalid"):
        build_external_complete(
            run_root=preflight_root,
            partition="test",
            selected_methods=["nninteractive"],
            comparator_config=external["comparator_config"],
            experiment_config=config,
            learning_split=split,
            input_manifest=missing_input,
            natural_episode_manifest=external["natural_episode_manifest"],
            runtime_receipts={"nninteractive": external["runtime"]},
            core_python=external["core_python"],
            official_metrics=external["official_metrics"],
            nninteractive_python=external["nninteractive_python"],
            test_access_receipt=invalid_receipt,
            frozen_external_admission=external["admission"],
            route_a_run_root=route_root,
            ledger_root=ledger,
            output=preflight_root / "artifacts" / "invalid-receipt.json",
        )

    wrong_runtime = route_root / "wrong-runtime.json"
    wrong_runtime.write_bytes(external["runtime"].read_bytes())
    with pytest.raises(ExternalCompleteError, match="runtime receipt differs"):
        build_external_complete(
            run_root=preflight_root,
            partition="test",
            selected_methods=["nninteractive"],
            comparator_config=external["comparator_config"],
            experiment_config=config,
            learning_split=split,
            input_manifest=missing_input,
            natural_episode_manifest=external["natural_episode_manifest"],
            runtime_receipts={"nninteractive": wrong_runtime},
            core_python=external["core_python"],
            official_metrics=external["official_metrics"],
            nninteractive_python=external["nninteractive_python"],
            test_access_receipt=test_receipt,
            frozen_external_admission=external["admission"],
            route_a_run_root=route_root,
            ledger_root=ledger,
            output=preflight_root / "artifacts" / "wrong-runtime.json",
        )

    wrong_core = _executable(route_root / "wrong-core-python", b"wrong-core")
    with pytest.raises(ExternalCompleteError, match="executable/metrics differ"):
        build_external_complete(
            run_root=preflight_root,
            partition="test",
            selected_methods=["nninteractive"],
            comparator_config=external["comparator_config"],
            experiment_config=config,
            learning_split=split,
            input_manifest=missing_input,
            natural_episode_manifest=external["natural_episode_manifest"],
            runtime_receipts={"nninteractive": external["runtime"]},
            core_python=wrong_core,
            official_metrics=external["official_metrics"],
            nninteractive_python=external["nninteractive_python"],
            test_access_receipt=test_receipt,
            frozen_external_admission=external["admission"],
            route_a_run_root=route_root,
            ledger_root=ledger,
            output=preflight_root / "artifacts" / "wrong-core.json",
        )
    wrong_metrics = route_root / "wrong-metrics.py"
    wrong_metrics.write_text("class MetricEvaluator: pass  # wrong\n", encoding="utf-8")
    with pytest.raises(ExternalCompleteError, match="executable/metrics differ"):
        build_external_complete(
            run_root=preflight_root,
            partition="test",
            selected_methods=["nninteractive"],
            comparator_config=external["comparator_config"],
            experiment_config=config,
            learning_split=split,
            input_manifest=missing_input,
            natural_episode_manifest=external["natural_episode_manifest"],
            runtime_receipts={"nninteractive": external["runtime"]},
            core_python=external["core_python"],
            official_metrics=wrong_metrics,
            nninteractive_python=external["nninteractive_python"],
            test_access_receipt=test_receipt,
            frozen_external_admission=external["admission"],
            route_a_run_root=route_root,
            ledger_root=ledger,
            output=preflight_root / "artifacts" / "wrong-metrics.json",
        )

    val_root = tmp_path / "external-val"
    test_root = route_root / "external-test"
    artifacts = test_root / "artifacts"
    metrics = test_root / "metrics"
    artifacts.mkdir(parents=True)
    metrics.mkdir()
    test_natural = test_root / "natural_episodes.jsonl"
    test_natural.write_text(
        json.dumps(
            {
                "episode_id": "episode-test",
                "case_id": "case-test",
                "patient_id": "patient-test",
                "partition": "test",
                "learning_split_sha256": _sha(split),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    input_manifest = artifacts / "external_comparator_input.json"
    input_manifest.write_text(
        json.dumps(
            {
                "schema_version": "PETCT-EXTERNAL-COMPARATOR-INPUT-v1.0",
                "status": "FROZEN_INPUT_READY",
                "partition": "test",
                "record_count": 1,
                "patient_count": 1,
                "provenance": {
                    "learning_split_sha256": _sha(split),
                    "natural_episode_manifest_sha256": _sha(test_natural),
                },
                "records": [
                    {
                        "case_id": "case-test",
                        "patient_id": "patient-test",
                        "split": "test",
                        "episode_id": "episode-test",
                        "patient_split_receipt": {
                            "internal_partition": "test",
                            "learning_split_sha256": _sha(split),
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    for policy in ("union_with_m0", "native_full_mask"):
        prediction_dir = test_root / "predictions" / f"nninteractive_{policy}"
        prediction_dir.mkdir(parents=True)
        prediction = prediction_dir / "case-test.nii.gz"
        prediction.write_bytes(
            (
                val_root
                / "predictions"
                / f"nninteractive_{policy}"
                / "case-1.nii.gz"
            ).read_bytes()
        )
        output = json.loads(
            (
                val_root / "artifacts" / f"nninteractive_{policy}_output.json"
            ).read_text(encoding="utf-8")
        )
        output["records"][0]["case_id"] = "case-test"
        output["records"][0]["patient_id"] = "patient-test"
        output["records"][0]["prediction_path"] = str(prediction.resolve())
        output["records"][0]["prediction_sha256"] = _sha(prediction)
        output["records"][0]["checkpoint_sha256"] = "0" * 64
        output["records"][0]["source_checkpoint_id"] = "nnInteractive_v1.0:" + "0" * 64
        output["test_access_receipt_sha256"] = _sha(test_receipt)
        output_path = artifacts / f"nninteractive_{policy}_output.json"
        output_path.write_text(json.dumps(output), encoding="utf-8")
        rows = val_root / "metrics" / f"nninteractive_{policy}_rows.jsonl"
        metric_row = json.loads(rows.read_text(encoding="utf-8"))
        metric_row["case_id"] = "case-test"
        metric_row["episode_id"] = "episode-test"
        metric_row["patient_id"] = "patient-test"
        metric_row["split"] = "test"
        (metrics / rows.name).write_text(
            json.dumps(metric_row) + "\n", encoding="utf-8"
        )
        summary = json.loads(
            (
                val_root / "metrics" / f"nninteractive_{policy}_summary.json"
            ).read_text(encoding="utf-8")
        )
        summary["input_manifest_sha256"] = _sha(input_manifest)
        summary["output_manifest_sha256"] = _sha(output_path)
        summary["test_access_receipt_sha256"] = _sha(test_receipt)
        summary["natural_episode_manifest_sha256"] = _sha(test_natural)
        summary["metric_rows_sha256"] = _sha(metrics / rows.name)
        summary["patient_clustered"] = {
            name: patient_cluster_summary([metric_row], name)
            for name in REQUIRED_METRICS
        }
        (metrics / f"nninteractive_{policy}_summary.json").write_text(
            json.dumps(summary), encoding="utf-8"
        )
    with pytest.raises(
        ExternalCompleteError, match="did not use the current admitted checkpoint"
    ):
        build_external_complete(
            run_root=test_root,
            partition="test",
            selected_methods=["nninteractive"],
            comparator_config=external["comparator_config"],
            experiment_config=config,
            learning_split=split,
            input_manifest=input_manifest,
            natural_episode_manifest=test_natural,
            runtime_receipts={"nninteractive": external["runtime"]},
            core_python=external["core_python"],
            official_metrics=external["official_metrics"],
            nninteractive_python=external["nninteractive_python"],
            test_access_receipt=test_receipt,
            frozen_external_admission=external["admission"],
            route_a_run_root=route_root,
            ledger_root=ledger,
            output=artifacts / "EXTERNAL_COMPARATORS_COMPLETE.json",
        )
