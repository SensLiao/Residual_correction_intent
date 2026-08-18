from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest


PROJECT = Path(__file__).resolve().parents[1]
MAINLINE_DATASET_ID = "R13-main-single-round"
MAINLINE_SOURCE = "M0_V6_FIVEFOLD_OOF"


def _lineage_module():
    import importlib.util

    path = PROJECT / "scripts" / "common" / "petct_mainline_lineage.py"
    assert path.is_file(), "R13 lineage validator is not implemented"
    spec = importlib.util.spec_from_file_location("petct_mainline_lineage", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _record(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": _sha(path),
    }


def _write_lineage(tmp_path: Path, **overrides: object) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    oof = tmp_path / "M0_V6_FIVEFOLD_OOF_READY.json"
    split = tmp_path / "learning_split.json"
    config = tmp_path / "experiment_v3.json"
    oof.write_text("{}\n", encoding="utf-8")
    split.write_text("{}\n", encoding="utf-8")
    config.write_text("{}\n", encoding="utf-8")
    payload: dict[str, object] = {
        "schema_version": "PETCT-R13-LINEAGE-v1.0",
        "status": "PASS",
        "dataset_id": MAINLINE_DATASET_ID,
        "source_m0_lineage": MAINLINE_SOURCE,
        "mainline_eligible": True,
        "lifecycle": "active",
        "episode_schema": "single_round_one_scribble_one_strategy_v1",
        "round_count": 1,
        "scribbles_per_episode": 1,
        "strategy_is_label": False,
        "partitions": ["train", "val"],
        "locked_test_present": False,
        "oof_ready": _record(oof),
        "learning_split": _record(split),
        "experiment_config": _record(config),
    }
    payload.update(overrides)
    receipt = tmp_path / "lineage-receipt.json"
    receipt.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return receipt


def test_r13_lineage_accepts_only_m0_v6_single_round_active_dataset(
    tmp_path: Path,
) -> None:
    lineage = _lineage_module()
    receipt = _write_lineage(tmp_path)
    validated = lineage.validate_r13_lineage_receipt(receipt)
    assert validated["source_m0_lineage"] == MAINLINE_SOURCE
    assert validated["dataset_id"] == MAINLINE_DATASET_ID

    legacy = _write_lineage(
        tmp_path / "legacy",
        source_m0_lineage="PETCT-M0-OOF-v1.0",
        lifecycle="deprecated",
        mainline_eligible=False,
    )
    with pytest.raises(lineage.LineageContractError, match="M0_V6_FIVEFOLD_OOF"):
        lineage.validate_r13_lineage_receipt(legacy)


def test_r13_program_rows_are_single_round_single_scribble_strategy_siblings() -> None:
    lineage = _lineage_module()
    rows = [
        {
            "episode_id": f"episode-{strategy}",
            "episode_family_id": "family-1",
            "partition": "train",
            "round_index": 0,
            "scribble_count": 1,
            "strategy": strategy,
            "source_m0_lineage": MAINLINE_SOURCE,
        }
        for strategy in ("centerline", "random", "boundary")
    ]
    lineage.validate_r13_program_rows(rows)

    too_many = rows + [{**rows[0], "episode_id": "episode-fourth"}]
    with pytest.raises(lineage.LineageContractError, match="at most three strategy siblings"):
        lineage.validate_r13_program_rows(too_many)

    trajectory = [{**rows[0], "round_index": 1}]
    with pytest.raises(lineage.LineageContractError, match="single-round"):
        lineage.validate_r13_program_rows(trajectory)


def test_active_launchers_are_explicitly_r13_bound_and_legacy_launchers_retire() -> None:
    r13 = PROJECT / "scripts" / "orchestration" / "launch_petct_r13_mainline.sh"
    text = r13.read_text(encoding="utf-8")
    assert "M0_V6_FIVEFOLD_OOF" in text
    assert "lineage-receipt.json" in text
    assert "data-ready.json" in text
    assert "--oof-ready" in text
    assert "latest" not in text.casefold()
    assert "glob" not in text.casefold()
    for forbidden in ("PETCT-TRAIN-20260805-R1", "PETCT-TRAIN-20260807-R1", "PETCT-MSLCP-GATE0-20260817-R12"):
        assert forbidden not in text

    for name in (
        "launch_petct_r12_pointer_targets.sh",
        "launch_petct_gate0b_determinism.sh",
        "launch_petct_gate0c_legacy_diagnostic.sh",
    ):
        retired = (PROJECT / "scripts" / "orchestration" / name).read_text(
            encoding="utf-8"
        )
        assert "HISTORICAL_ONLY" in retired
        assert "exit 64" in retired


def test_v3_trainers_require_a_lineage_receipt() -> None:
    for relative in (
        "scripts/p2t/train_petct_program_v3.py",
        "scripts/editor/train_petct_program_editor_v3.py",
    ):
        text = (PROJECT / relative).read_text(encoding="utf-8")
        assert '"--lineage-receipt"' in text
        assert "validate_r13_training_binding" in text


def test_flat_p2t_baseline_train_and_eval_require_r13_data_ready() -> None:
    for relative in (
        "scripts/p2t/train_petct_p2t.py",
        "scripts/evaluation/evaluate_petct_p2t.py",
    ):
        text = (PROJECT / relative).read_text(encoding="utf-8")
        assert '"--r13-data-ready"' in text
        assert "validate_r13_data_ready" in text


def test_oof_wait_queue_never_shares_or_kills_a_gpu() -> None:
    queue = (
        PROJECT
        / "scripts"
        / "orchestration"
        / "launch_petct_m0_v6_oof_when_free.sh"
    ).read_text(encoding="utf-8")
    assert "safe_gpu_check_3_of_3" in queue
    assert "--query-compute-apps=gpu_uuid" in queue
    assert "temperature.gpu" in queue
    assert "kill" not in queue.casefold()
    assert "M0_V6_FIVEFOLD_OOF" in queue


def test_oof_queue_uses_any_free_card_and_runner_accepts_single_gpu() -> None:
    queue = (
        PROJECT
        / "scripts"
        / "orchestration"
        / "launch_petct_m0_v6_oof_when_free.sh"
    ).read_text(encoding="utf-8")
    assert '"free_gpus":"%s"' in queue
    assert "LAUNCH_GPU1" in queue
    runner = (
        PROJECT
        / "scripts"
        / "orchestration"
        / "launch_petct_m0_v6_oof.sh"
    ).read_text(encoding="utf-8")
    assert 'run_queue "$GPU0" 0 1 2 3 4' in runner
    assert "gpu1=-1 for single-GPU mode" in runner


def test_z390_w21_edt_fold234_queue_is_single_gpu_and_test_free() -> None:
    text = (
        PROJECT
        / "scripts"
        / "orchestration"
        / "launch_w21_edt_fold234_when_free_z390.sh"
    ).read_text(encoding="utf-8")
    assert "W21_EDT_FOLD234_DONE.json" in text
    assert "--query-compute-apps=gpu_uuid" in text
    assert "temperature.gpu" in text
    assert "safe_gpu_check_3_of_3" in text
    assert "eb56870cd52af383" in text  # five-fold split pin
    assert "for fold in 2 3 4" in text
    assert "nnUNetv2_train 903 3d_fullres" in text
    assert "kill" not in text.casefold()
    assert "--npz" in text
    assert "CUDA_VISIBLE_DEVICES=\"$GPU\"" in text


def test_superseded_route_and_d3_orchestrators_fail_before_any_stage() -> None:
    for name in (
        "run_petct_route_a_after_baseline.sh",
        "watch_and_run_petct_route_a_after_m0.sh",
        "run_petct_legacy_m0_d3_generation.sh",
        "run_petct_legacy_m0_d3_rematerialize.sh",
    ):
        prefix = "\n".join(
            (
                PROJECT / "scripts" / "orchestration" / name
            ).read_text(encoding="utf-8").splitlines()[:10]
        )
        assert "HISTORICAL_ONLY" in prefix
        assert "exit 64" in prefix


def test_r13_config_inherits_frozen_data_construction_values() -> None:
    v2 = json.loads(
        (PROJECT / "configs" / "petct_route_a_experiment.json").read_text(
            encoding="utf-8"
        )
    )
    r13 = json.loads(
        (PROJECT / "configs" / "petct_route_a_experiment_v3.json").read_text(
            encoding="utf-8"
        )
    )
    for key in ("dataset", "learning_tensor_normalization", "p2t"):
        assert r13[key] == v2[key]
    for key in (
        "source",
        "commit",
        "simulator_file_sha256",
        "strategies",
        "minimum_best_slice_pixels",
        "primary_strategy_mode",
        "primary_strategy_salt",
        "seed",
    ):
        assert r13["scribble"][key] == v2["scribble"][key]
    assert r13["editor"]["local_radius_mm"] == v2["editor"]["local_radius_mm"]
    assert r13["editor"]["minimum_local_area_mm2"] == v2["editor"]["minimum_local_area_mm2"]


def test_r13_smoke_launcher_covers_every_effect_architecture_and_predicted_chain() -> None:
    text = (
        PROJECT
        / "scripts"
        / "orchestration"
        / "launch_petct_r13_effect_smoke.sh"
    ).read_text(encoding="utf-8")
    for arm in ("J1", "J2", "J6", "J7", "J8", "J9", "J9C"):
        assert arm in text
    assert "data-ready.json" in text
    assert "petct_mainline_lineage.py" in text
    assert "infer_petct_program_v3.py" in text
    assert "infer_petct_program_editor_v3.py" in text
    assert "evaluate_petct_program_v3.py" in text
    assert "cmp -s" in text
    assert "gate0b.done" in text
    assert "gate0c.done" in text
    assert "--partition val" in text
    assert "--partition test" not in text
    assert "latest" not in text.casefold()
    assert "glob" not in text.casefold()


def test_r13_full_val_launcher_uses_frozen_budgets_and_no_test() -> None:
    text = (
        PROJECT
        / "scripts"
        / "orchestration"
        / "launch_petct_r13_effect_val.sh"
    ).read_text(encoding="utf-8")
    for arm in ("J1", "J2", "J6", "J7", "J8", "J9", "J9C"):
        assert arm in text
    assert "--partition val" in text
    assert "--partition test" not in text
    assert "--smoke-one-epoch" not in text
    assert "--epochs 1" not in text
    assert "data-ready.json" in text
    assert "M0_V6_FIVEFOLD_OOF" in text


def test_r13_pipeline_watcher_orders_oof_data_smoke_then_full_val() -> None:
    text = (
        PROJECT
        / "scripts"
        / "orchestration"
        / "watch_petct_r13_pipeline.sh"
    ).read_text(encoding="utf-8")
    oof = text.index("M0_V6_FIVEFOLD_OOF_READY.json")
    data = text.index("launch_petct_r13_mainline.sh")
    smoke = text.index("launch_petct_r13_effect_smoke.sh")
    full = text.index("launch_petct_r13_effect_val.sh")
    assert oof < data < smoke < full
    assert "safe_gpu_check_3_of_3" in text
    assert "--partition test" not in text
    assert "locked TEST" in text


def test_cleanup_watcher_waits_for_full_val_before_exact_cleanup_confirmation() -> None:
    text = (
        PROJECT
        / "scripts"
        / "orchestration"
        / "watch_petct_r13_cleanup.sh"
    ).read_text(encoding="utf-8")
    assert "effect_val.done" in text
    assert "cleanup_petct_legacy_after_r13.py" in text
    assert "DELETE_SUPERSEDED_R13_LEGACY" in text
    assert text.index("effect_val.done") < text.index('"$PY" "$CLEANUP"')


def test_inference_row_allowlist_accepts_r13_identity_and_candidate_fields():
    import importlib.util
    import sys

    path = PROJECT / "scripts" / "common" / "petct_program_learning.py"
    spec = importlib.util.spec_from_file_location(
        "petct_program_learning_r13_contract", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    if str(PROJECT / "scripts") not in sys.path:
        sys.path.insert(0, str(PROJECT / "scripts"))
    spec.loader.exec_module(module)
    row = {
        "schema_version": "PETCT-PROGRAM-INFERENCE-MANIFEST-v1.0",
        "episode_id": "ep-a",
        "partition": "train",
        "operation": "ADD",
        "visible_npz": "/visible/ep-a.npz",
        "visible_sha256": "a" * 64,
        "candidate_json": "/visible/candidates/ep-a.json",
        "candidate_sha256": "b" * 64,
        "dataset_id": MAINLINE_DATASET_ID,
        "source_m0_lineage": MAINLINE_SOURCE,
        "round_index": 0,
        "scribble_count": 1,
    }
    module._validate_inference_row(row)
