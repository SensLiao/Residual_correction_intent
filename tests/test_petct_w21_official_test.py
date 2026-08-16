from __future__ import annotations

import __future__
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import nibabel as nib

PROJECT = Path(__file__).resolve().parents[1]
SCRIPTS = PROJECT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

OFFICIAL_SIMULATOR = (
    PROJECT / "upstream" / "autoPETV" / "interactive" / "simulate_scribbles.py"
)
OFFICIAL_METRICS = PROJECT / "upstream" / "autoPETV" / "metrics.py"
requires_official_runtime = pytest.mark.skipif(
    not (OFFICIAL_SIMULATOR.is_file() and OFFICIAL_METRICS.is_file()),
    reason="pinned autoPETV simulator and metrics runtime is not bundled here",
)

import common.petct_w21_test_access as access  # noqa: E402
import evaluation.run_petct_w21_official_test as w21_runner  # noqa: E402
from common.petct_w21_test_access import (  # noqa: E402
    CONFIRMATION,
    W21AccessError,
    assign_case_level_strategies,
    build_clean_learning_inventory,
    build_test_inventory,
    consume_grant,
    create_grant,
    validate_receipt,
)
from evaluation.run_petct_w21_official_test import (  # noqa: E402
    choose_correction,
    compute_state_metrics,
    encode_edt,
    per_case_auc,
)


def _write_contract(tmp_path: Path) -> tuple[Path, Path]:
    patients = []
    identity = []
    case_number = 0
    for patient_number in range(378):
        patient_id = f"patient-{patient_number:03d}"
        if patient_number < 57:
            partition = "test"
            case_count = 2 if patient_number < 34 else 1
        else:
            partition = "train"
            case_count = 2 if patient_number - 57 < 185 else 1
        case_ids = []
        for _ in range(case_count):
            case_id = f"case-{case_number:04d}"
            case_number += 1
            case_ids.append(case_id)
            identity.append(
                {
                    "case_id": case_id,
                    "patient_id": patient_id,
                    "held_out_fold": patient_number % 5,
                    "ct_path": f"/data/{case_id}_0000.nii.gz",
                    "pet_path": f"/data/{case_id}_0001.nii.gz",
                    "gt_path": f"/data/{case_id}.nii.gz",
                    "truth_materialization": "IDENTITY_ONLY",
                }
            )
        patients.append(
            {
                "patient_id": patient_id,
                "partition": partition,
                "case_ids": case_ids,
                "rank_sha256": "0" * 64,
            }
        )
    assert case_number == 597
    identity_path = tmp_path / "identity.jsonl"
    identity_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in identity),
        encoding="utf-8",
    )
    split_path = tmp_path / "learning_split.json"
    split_path.write_text(
        json.dumps(
            {
                "schema_version": "PETCT-LEARNING-SPLIT-v1.0",
                "status": "FROZEN_BEFORE_MODEL_SELECTION",
                "case_count": 597,
                "patient_count": 378,
                "case_counts": {"train": 506, "val": 0, "test": 91},
                "patients": patients,
            }
        ),
        encoding="utf-8",
    )
    return identity_path, split_path


def _model(
    tmp_path: Path,
    arm: str,
    *,
    learning_split: Path,
    clean_inventory: dict,
) -> Path:
    root = tmp_path / arm
    (root / "fold_0").mkdir(parents=True)
    (root / "dataset.json").write_text("{}", encoding="utf-8")
    (root / "plans.json").write_text("{}", encoding="utf-8")
    checkpoint = root / "fold_0" / "checkpoint_final.pth"
    checkpoint.write_bytes(arm.encode("ascii"))
    (root / "training_provenance.json").write_text(
        json.dumps(
            {
                "schema_version": access.MODEL_PROVENANCE_SCHEMA,
                "arm": arm,
                "learning_split_sha256": access.sha256_file(learning_split),
                "clean_train_val_case_inventory_sha256": clean_inventory[
                    "case_inventory_sha256"
                ],
                "clean_train_val_patient_inventory_sha256": clean_inventory[
                    "patient_inventory_sha256"
                ],
                "training_partitions": ["train", "val"],
                "training_case_count": 506,
                "training_patient_count": 321,
                "test_case_count_consumed": 0,
                "fold": access.EXPECTED_MODEL_FOLD,
                "checkpoint_name": access.EXPECTED_CHECKPOINT_NAME,
                "checkpoint_sha256": access.sha256_file(checkpoint),
                "checkpoint_selection": access.EXPECTED_CHECKPOINT_SELECTION,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return root


@requires_official_runtime
def test_access_grant_consumption_and_receipt_are_hash_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    identity, split = _write_contract(tmp_path)
    monkeypatch.setattr(
        access, "EXPECTED_LEARNING_SPLIT_SHA256", access.sha256_file(split)
    )
    inventory = build_test_inventory(identity, split)
    clean_inventory = build_clean_learning_inventory(split)
    assert inventory["case_count"] == 91
    assert inventory["patient_count"] == 57
    assert sorted(inventory["strategy_case_counts"].values()) == [30, 30, 31]
    assert {row["strategy"] for row in inventory["cases"]} == {
        "centerline",
        "random",
        "boundary",
    }
    run_root = tmp_path / "run"
    ledger_root = tmp_path / "ledger"
    run_root.mkdir()
    ledger_root.mkdir()
    grant_path = run_root / "grant.json"
    create_grant(
        identity_manifest=identity,
        learning_split=split,
        binary_model_dir=_model(
            tmp_path,
            "binary",
            learning_split=split,
            clean_inventory=clean_inventory,
        ),
        edt_model_dir=_model(
            tmp_path,
            "edt",
            learning_split=split,
            clean_inventory=clean_inventory,
        ),
        runner_script=PROJECT
        / "scripts"
        / "evaluation"
        / "run_petct_w21_official_test.py",
        simulator_script=PROJECT
        / "upstream"
        / "autoPETV"
        / "interactive"
        / "simulate_scribbles.py",
        metric_script=PROJECT / "upstream" / "autoPETV" / "metrics.py",
        access_script=PROJECT / "scripts" / "common" / "petct_w21_test_access.py",
        run_root=run_root,
        ledger_root=ledger_root,
        grant_path=grant_path,
        authorized_by="director-delegated-codex",
        confirmation=CONFIRMATION,
    )
    grant = json.loads(grant_path.read_text(encoding="utf-8"))
    official = grant["binding"]["code"]["official_autopetv"]
    assert official["commit"] == access.OFFICIAL_AUTOPETV_COMMIT
    assert official["metrics"]["sha256"] == access.OFFICIAL_METRICS_SHA256
    assert official["simulator"]["sha256"] == access.OFFICIAL_SIMULATOR_SHA256
    assert grant["binding"]["clean_learning_inventory"] == clean_inventory
    receipt_path = run_root / "receipt.json"
    receipt = consume_grant(grant_path=grant_path, receipt_path=receipt_path)
    assert validate_receipt(receipt_path)["receipt_sha256"] == receipt["receipt_sha256"]
    with pytest.raises(W21AccessError, match="already consumed"):
        consume_grant(grant_path=grant_path, receipt_path=run_root / "second.json")


def test_case_level_strategy_allocator_is_deterministic_and_exactly_balanced() -> None:
    case_ids = [f"case-{index:03d}" for index in range(91)]
    first = assign_case_level_strategies(case_ids)
    second = assign_case_level_strategies(list(reversed(case_ids)))
    assert first == second
    assert sorted(first["strategy_case_counts"].values()) == [30, 30, 31]
    assert "not-upstream-verbatim" in access.PROTOCOL[
        "strategy_assignment_relation"
    ]


@requires_official_runtime
def test_formal_grant_rejects_unpinned_official_code_before_consumption(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    identity, split = _write_contract(tmp_path)
    monkeypatch.setattr(
        access, "EXPECTED_LEARNING_SPLIT_SHA256", access.sha256_file(split)
    )
    clean_inventory = build_clean_learning_inventory(split)
    fake_metrics = tmp_path / "metrics.py"
    fake_metrics.write_text("not official", encoding="utf-8")
    run_root = tmp_path / "run"
    ledger_root = tmp_path / "ledger"
    run_root.mkdir()
    ledger_root.mkdir()
    with pytest.raises(W21AccessError, match="official metrics hash"):
        create_grant(
            identity_manifest=identity,
            learning_split=split,
            binary_model_dir=_model(
                tmp_path,
                "binary",
                learning_split=split,
                clean_inventory=clean_inventory,
            ),
            edt_model_dir=_model(
                tmp_path,
                "edt",
                learning_split=split,
                clean_inventory=clean_inventory,
            ),
            runner_script=PROJECT
            / "scripts"
            / "evaluation"
            / "run_petct_w21_official_test.py",
            simulator_script=PROJECT
            / "upstream"
            / "autoPETV"
            / "interactive"
            / "simulate_scribbles.py",
            metric_script=fake_metrics,
            access_script=PROJECT
            / "scripts"
            / "common"
            / "petct_w21_test_access.py",
            run_root=run_root,
            ledger_root=ledger_root,
            grant_path=run_root / "grant.json",
            authorized_by="director-delegated-codex",
            confirmation=CONFIRMATION,
        )


def test_formal_grant_rejects_noncanonical_learning_split_hash(tmp_path: Path) -> None:
    identity, split = _write_contract(tmp_path)
    run_root = tmp_path / "run"
    ledger_root = tmp_path / "ledger"
    run_root.mkdir()
    ledger_root.mkdir()
    with pytest.raises(W21AccessError, match="canonical clean split"):
        create_grant(
            identity_manifest=identity,
            learning_split=split,
            binary_model_dir=tmp_path / "unused-binary",
            edt_model_dir=tmp_path / "unused-edt",
            runner_script=PROJECT
            / "scripts"
            / "evaluation"
            / "run_petct_w21_official_test.py",
            simulator_script=PROJECT
            / "upstream"
            / "autoPETV"
            / "interactive"
            / "simulate_scribbles.py",
            metric_script=PROJECT / "upstream" / "autoPETV" / "metrics.py",
            access_script=PROJECT
            / "scripts"
            / "common"
            / "petct_w21_test_access.py",
            run_root=run_root,
            ledger_root=ledger_root,
            grant_path=run_root / "grant.json",
            authorized_by="director-delegated-codex",
            confirmation=CONFIRMATION,
        )


def test_formal_grant_rejects_wrong_fold_checkpoint_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    identity, split = _write_contract(tmp_path)
    monkeypatch.setattr(
        access, "EXPECTED_LEARNING_SPLIT_SHA256", access.sha256_file(split)
    )
    clean_inventory = build_clean_learning_inventory(split)
    binary = _model(
        tmp_path,
        "binary",
        learning_split=split,
        clean_inventory=clean_inventory,
    )
    provenance_path = binary / "training_provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["fold"] = 1
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")
    run_root = tmp_path / "run"
    ledger_root = tmp_path / "ledger"
    run_root.mkdir()
    ledger_root.mkdir()
    with pytest.raises(W21AccessError, match="training provenance"):
        create_grant(
            identity_manifest=identity,
            learning_split=split,
            binary_model_dir=binary,
            edt_model_dir=_model(
                tmp_path,
                "edt",
                learning_split=split,
                clean_inventory=clean_inventory,
            ),
            runner_script=PROJECT
            / "scripts"
            / "evaluation"
            / "run_petct_w21_official_test.py",
            simulator_script=PROJECT
            / "upstream"
            / "autoPETV"
            / "interactive"
            / "simulate_scribbles.py",
            metric_script=PROJECT / "upstream" / "autoPETV" / "metrics.py",
            access_script=PROJECT
            / "scripts"
            / "common"
            / "petct_w21_test_access.py",
            run_root=run_root,
            ledger_root=ledger_root,
            grant_path=run_root / "grant.json",
            authorized_by="director-delegated-codex",
            confirmation=CONFIRMATION,
        )


class _Simulator:
    @staticmethod
    def simulate_scribble_from_label(mask, strategy, seed):
        coordinates = np.argwhere(mask > 0)
        if not len(coordinates):
            return [], False  # the pinned official empty-residual bug
        selected = [[int(value) for value in coordinates[0]]]
        return selected, True, 1


def test_official_polarity_handles_empty_candidate_without_crashing() -> None:
    gt = np.zeros((4, 4, 4), dtype=np.uint8)
    gt[1, 1, 1] = 1
    foreground = choose_correction(
        np.zeros_like(gt), gt, strategy="centerline", simulator=_Simulator()
    )
    assert foreground["polarity"] == "foreground"
    prediction = np.zeros_like(gt)
    prediction[1, 1, 1] = 1
    prediction[2, 2, 2] = 1
    background = choose_correction(
        prediction, gt, strategy="centerline", simulator=_Simulator()
    )
    assert background["polarity"] == "background"


def test_edt_is_zero_for_no_scribble_and_bounded_for_one_voxel() -> None:
    empty = np.zeros((9, 9, 9), dtype=np.uint8)
    assert np.count_nonzero(encode_edt(empty)) == 0
    empty[4, 4, 4] = 1
    encoded = encode_edt(empty)
    assert encoded.dtype == np.float32
    assert encoded[4, 4, 4] == pytest.approx(1.0)
    assert 0.0 <= float(encoded.min()) <= float(encoded.max()) <= 1.0
    assert np.count_nonzero(encoded) > 1


def test_six_state_auc_is_unormalized_trapezoid() -> None:
    states = [{"dice": 0.8, "dmm_f1": 0.6} for _ in range(6)]
    auc = per_case_auc(states)
    assert auc == pytest.approx(
        {"auc_dice": 4.0, "auc_dmm": 3.0, "combined_score": 3.5}
    )


def _official_metric_evaluator():
    """Load the exact pinned source under the local Python 3.9 test runtime.

    The production environment is Python 3.10. Python 3.9 needs postponed
    annotations for the upstream ``int | tuple[...]`` annotation syntax; the
    compiler flag changes annotation evaluation only, not metric behavior.
    """

    source = OFFICIAL_METRICS.read_text(encoding="utf-8")
    module = types.ModuleType("test_w21_official_metrics")
    module.__file__ = str(OFFICIAL_METRICS)
    code = compile(
        source,
        str(OFFICIAL_METRICS),
        "exec",
        flags=__future__.annotations.compiler_flag,
        dont_inherit=True,
    )
    exec(code, module.__dict__)
    return module.MetricEvaluator


@requires_official_runtime
def test_empty_gt_is_excluded_from_auc_but_retains_official_and_raw_fpv() -> None:

    pred = np.zeros((2, 2, 2), dtype=np.uint8)
    pred[0, 0, 0] = 1
    metrics = compute_state_metrics(
        pred,
        np.zeros_like(pred),
        spacing=(2.0, 2.0, 2.0),
        metric_evaluator_class=_official_metric_evaluator(),
        case_name="empty",
    )
    assert metrics["dice"] is None
    assert metrics["dmm_f1"] is None
    assert metrics["fpv_ml"] == pytest.approx(0.008)
    assert metrics["fnv_ml"] is None
    assert metrics["raw_voxel_fp_volume_ml"] == pytest.approx(0.008)
    assert metrics["raw_voxel_fn_volume_ml"] == pytest.approx(0.0)


@requires_official_runtime
def test_raw_voxel_fp_is_not_mislabeled_as_official_component_fpv() -> None:
    gt = np.zeros((4, 4, 4), dtype=np.uint8)
    gt[1, 1, 1] = 1
    pred = gt.copy()
    pred[1, 1, 2] = 1
    metrics = compute_state_metrics(
        pred,
        gt,
        spacing=(1.0, 1.0, 1.0),
        metric_evaluator_class=_official_metric_evaluator(),
        case_name="one-matched-component-with-extra-voxel",
    )
    assert metrics["dmm_f1"] == pytest.approx(1.0)
    assert metrics["fpv_ml"] == pytest.approx(0.0)
    assert metrics["fnv_ml"] == pytest.approx(0.0)
    assert metrics["raw_voxel_fp_volume_ml"] == pytest.approx(0.001)
    assert metrics["raw_voxel_fn_volume_ml"] == pytest.approx(0.0)


def test_smoke_has_no_real_data_path_arguments_and_builds_only_synthetic_inputs(
    tmp_path: Path,
) -> None:
    parser = w21_runner._parser()
    subparsers = parser._subparsers._group_actions[0].choices
    smoke_options = {
        option
        for action in subparsers["smoke"]._actions
        for option in action.option_strings
    }
    assert {"--ct", "--pet", "--gt", "--case-id", "--patient-id"}.isdisjoint(
        smoke_options
    )

    source = w21_runner._build_synthetic_smoke_source(tmp_path)
    assert source["case_id"] == w21_runner.SYNTHETIC_SMOKE_CASE_ID
    assert source["data_scope"] == "SYNTHETIC_NO_REAL_SUBJECT"
    for key in ("ct_path", "pet_path", "gt_path"):
        assert Path(source[key]).is_relative_to(tmp_path)
    gt = np.asanyarray(nib.load(source["gt_path"]).dataobj)
    assert gt.shape == w21_runner.SYNTHETIC_SMOKE_SHAPE
    assert set(np.unique(gt)) == {0, 1}


@requires_official_runtime
def test_smoke_rejects_arbitrary_metric_or_simulator_code(tmp_path: Path) -> None:
    fake = tmp_path / "fake.py"
    fake.write_text("raise RuntimeError('must never execute')", encoding="utf-8")
    official_simulator = (
        PROJECT
        / "upstream"
        / "autoPETV"
        / "interactive"
        / "simulate_scribbles.py"
    )
    official_metrics = PROJECT / "upstream" / "autoPETV" / "metrics.py"
    with pytest.raises(w21_runner.W21RunError, match="smoke simulator"):
        w21_runner._validate_pinned_official_smoke_code(fake, official_metrics)
    with pytest.raises(w21_runner.W21RunError, match="smoke metrics"):
        w21_runner._validate_pinned_official_smoke_code(official_simulator, fake)
    assert w21_runner._validate_pinned_official_smoke_code(
        official_simulator, official_metrics
    ) == (official_simulator.resolve(), official_metrics.resolve())
