from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest


PROJECT = Path(__file__).resolve().parents[1]
SCRIPTS = PROJECT / "scripts"
sys.path.insert(0, str(SCRIPTS))

from common.petct_test_access import (  # noqa: E402
    FINAL_FREEZE_CONFIRMATION,
    TestAccessError,
    _canonical_sha256,
    _validate_config_and_split,
    consume_final_freeze_grant,
    create_final_freeze_grant,
    enforce_partition_access,
    validate_consumed_receipt,
)
from common.petct_development_freeze import build_final_development_freeze  # noqa: E402
from test_petct_development_freeze import (  # noqa: E402
    _artifact_manifest,
    _config_and_split,
)


def test_owned_leaf_clis_use_consumed_receipts_not_boolean_unlocks() -> None:
    paths = (
        "scripts/data/build_petct_residual_manifest.py",
        "scripts/p2t/build_petct_matched_state_dataset.py",
        "scripts/evaluation/evaluate_petct_p2t.py",
        "scripts/editor/build_petct_intent_interventions.py",
        "scripts/editor/infer_petct_residual_editor.py",
        "scripts/evaluation/evaluate_petct_correction.py",
    )
    for relative in paths:
        source = (PROJECT / relative).read_text(encoding="utf-8")
        assert "--allow-test-access" not in source
        assert "enforce_partition_access(" in source
        assert "add_leaf_test_access_arguments(parser)" in source


def _write_contracts(root: Path) -> tuple[Path, Path]:
    return _config_and_split(root, test_policy=True)


def _grant(root: Path, config: Path, split: Path) -> Path:
    manifest, _ = _artifact_manifest(root, config, split)
    freeze = root / "FINAL_DEVELOPMENT_FREEZE.json"
    build_final_development_freeze(
        experiment_config=config,
        learning_split=split,
        required_artifacts_manifest=manifest,
        output=freeze,
    )
    grant = root / "FINAL_FREEZE_GRANT.json"
    create_final_freeze_grant(
        experiment_config=config,
        learning_split=split,
        final_development_freeze=freeze,
        grant_path=grant,
        authorized_by="codex-on-director-authority",
        confirmation=FINAL_FREEZE_CONFIRMATION,
        now=lambda: "2026-07-18T00:00:00Z",
    )
    return grant


def test_production_v2_test_access_policy_pair_is_accepted(tmp_path: Path) -> None:
    config, split = _write_contracts(tmp_path)
    document = json.loads(config.read_text(encoding="utf-8"))
    document["schema_version"] = "PETCT-ROUTE-A-EXPERIMENT-v2.0"
    document["dataset"]["learning_split"]["test_access"] = (
        "exactly-once-after-all-v2-development-freezes"
    )
    document["statistics"]["test_access"] = "exactly-once-after-v2-freeze"
    config.write_text(json.dumps(document), encoding="utf-8")

    _, _, config_record, split_record = _validate_config_and_split(config, split)

    assert config_record["path"] == str(config.resolve())
    assert split_record["path"] == str(split.resolve())


def test_production_v3_test_access_policy_pair_is_accepted(tmp_path: Path) -> None:
    """The sealed v3 config carries the v2 development policy string and a
    None statistics policy (statistics rides the same final-freeze grant).
    Registering exactly this pair unblocks the grant path (MEDIUM-1 fix)."""

    config, split = _write_contracts(tmp_path)
    document = json.loads(config.read_text(encoding="utf-8"))
    document["schema_version"] = "PETCT-ROUTE-A-EXPERIMENT-v3.0"
    document["dataset"]["learning_split"]["test_access"] = (
        "exactly-once-after-all-v2-development-freezes"
    )
    document["statistics"]["test_access"] = None
    config.write_text(json.dumps(document), encoding="utf-8")

    _, _, config_record, split_record = _validate_config_and_split(config, split)

    assert config_record["path"] == str(config.resolve())
    assert split_record["path"] == str(split.resolve())


def test_v3_policy_rejects_wrong_statistics_string(tmp_path: Path) -> None:
    config, split = _write_contracts(tmp_path)
    document = json.loads(config.read_text(encoding="utf-8"))
    document["schema_version"] = "PETCT-ROUTE-A-EXPERIMENT-v3.0"
    document["dataset"]["learning_split"]["test_access"] = (
        "exactly-once-after-all-v2-development-freezes"
    )
    document["statistics"]["test_access"] = "exactly-once-after-v2-freeze"
    config.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(TestAccessError, match="test-access policy"):
        _validate_config_and_split(config, split)


@pytest.mark.parametrize(
    ("development_policy", "statistics_policy"),
    (
        (
            "exactly-once-after-all-development-freezes",
            "exactly-once-after-v2-freeze",
        ),
        (
            "exactly-once-after-all-v2-development-freezes",
            "exactly-once-after-freeze",
        ),
    ),
)
def test_v2_policy_rejects_old_and_new_mixed_pairs(
    tmp_path: Path,
    development_policy: str,
    statistics_policy: str,
) -> None:
    config, split = _write_contracts(tmp_path)
    document = json.loads(config.read_text(encoding="utf-8"))
    document["schema_version"] = "PETCT-ROUTE-A-EXPERIMENT-v2.0"
    document["dataset"]["learning_split"]["test_access"] = development_policy
    document["statistics"]["test_access"] = statistics_policy
    config.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(TestAccessError, match="test-access policy"):
        _validate_config_and_split(config, split)


def test_grant_rejects_checkpoint_changed_after_final_freeze(tmp_path: Path) -> None:
    config, split = _write_contracts(tmp_path)
    grant = _grant(tmp_path, config, split)
    checkpoint = tmp_path / "p2t-primary.pth"
    checkpoint.write_bytes(b"changed-after-freeze")
    run = tmp_path / "run"
    run.mkdir()
    with pytest.raises(TestAccessError, match="final development freeze is invalid"):
        consume_final_freeze_grant(
            grant_path=grant,
            run_root=run,
            receipt_path=run / "receipt.json",
            ledger_root=tmp_path / "ledger",
        )


def test_exactly_one_concurrent_global_consumption_wins(tmp_path: Path) -> None:
    config, split = _write_contracts(tmp_path)
    grant = _grant(tmp_path, config, split)
    ledger = tmp_path / "global-ledger"
    run_a, run_b = tmp_path / "run-a", tmp_path / "run-b"
    run_a.mkdir()
    run_b.mkdir()

    def consume(run: Path):
        try:
            return consume_final_freeze_grant(
                grant_path=grant,
                run_root=run,
                receipt_path=run / "governance" / "TEST_ACCESS_CONSUMED.json",
                ledger_root=ledger,
                now=lambda: "2026-07-18T00:01:00Z",
            )
        except TestAccessError as error:
            return error

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(consume, (run_a, run_b)))
    assert sum(isinstance(result, dict) for result in results) == 1
    assert sum(isinstance(result, TestAccessError) for result in results) == 1
    assert len(list(ledger.glob("*.json"))) == 1
    assert "already consumed globally" in str(
        next(result for result in results if isinstance(result, TestAccessError))
    )


def test_consumed_receipt_binds_config_split_partition_run_and_outputs(
    tmp_path: Path,
) -> None:
    config, split = _write_contracts(tmp_path)
    grant = _grant(tmp_path, config, split)
    ledger = tmp_path / "global-ledger"
    run = tmp_path / "run"
    run.mkdir()
    receipt_path = run / "governance" / "TEST_ACCESS_CONSUMED.json"
    receipt = consume_final_freeze_grant(
        grant_path=grant,
        run_root=run,
        receipt_path=receipt_path,
        ledger_root=ledger,
        now=lambda: "2026-07-18T00:01:00Z",
    )
    validated = validate_consumed_receipt(
        receipt_path,
        experiment_config=config,
        learning_split=split,
        run_root=run,
        ledger_root=ledger,
    )
    assert validated["receipt_sha256"] == receipt["receipt_sha256"]
    assert (
        validated["consumption"]["checkpoint_inventory_sha256"]
        == receipt["consumption"]["checkpoint_inventory_sha256"]
    )
    assert validated["consumption"]["allowed_partition"] == "test"
    assert validated["consumption"]["run_root"] == str(run.resolve())
    assert enforce_partition_access(
        "test",
        receipt_path=receipt_path,
        experiment_config=config,
        learning_split=split,
        run_root=run,
        output_paths=(run / "metrics" / "test.json",),
        ledger_root=ledger,
    )["receipt_sha256"] == receipt["receipt_sha256"]
    with pytest.raises(TestAccessError, match="escapes"):
        enforce_partition_access(
            "test",
            receipt_path=receipt_path,
            experiment_config=config,
            learning_split=split,
            run_root=run,
            output_paths=(tmp_path / "outside.json",),
            ledger_root=ledger,
        )
    with pytest.raises(TestAccessError, match="rejects a test receipt"):
        enforce_partition_access(
            "val",
            receipt_path=receipt_path,
            experiment_config=config,
            learning_split=split,
            run_root=run,
            ledger_root=ledger,
        )


def test_resealed_receipt_tamper_is_rejected_by_global_ledger(tmp_path: Path) -> None:
    config, split = _write_contracts(tmp_path)
    grant = _grant(tmp_path, config, split)
    ledger = tmp_path / "global-ledger"
    run = tmp_path / "run"
    other = tmp_path / "other-run"
    run.mkdir()
    other.mkdir()
    receipt_path = run / "TEST_ACCESS_CONSUMED.json"
    consume_final_freeze_grant(
        grant_path=grant,
        run_root=run,
        receipt_path=receipt_path,
        ledger_root=ledger,
    )
    forged = json.loads(receipt_path.read_text(encoding="utf-8"))
    forged["consumption"]["run_root"] = str(other.resolve())
    forged["consumption"]["receipt_path"] = str(receipt_path.resolve())
    forged["consumption_sha256"] = _canonical_sha256(forged["consumption"])
    unsigned = {key: value for key, value in forged.items() if key != "receipt_sha256"}
    forged["receipt_sha256"] = _canonical_sha256(unsigned)
    receipt_path.write_text(json.dumps(forged), encoding="utf-8")
    with pytest.raises(TestAccessError, match="global consumption ledger differs"):
        validate_consumed_receipt(
            receipt_path,
            experiment_config=config,
            learning_split=split,
            run_root=other,
            ledger_root=ledger,
        )


@pytest.mark.parametrize("target", ["config", "split"])
def test_config_or_split_mutation_invalidates_grant_before_consumption(
    tmp_path: Path, target: str
) -> None:
    config, split = _write_contracts(tmp_path)
    grant = _grant(tmp_path, config, split)
    path = config if target == "config" else split
    document = json.loads(path.read_text(encoding="utf-8"))
    document["tampered"] = True
    path.write_text(json.dumps(document), encoding="utf-8")
    run = tmp_path / "run"
    run.mkdir()
    with pytest.raises(TestAccessError, match="changed after grant"):
        consume_final_freeze_grant(
            grant_path=grant,
            run_root=run,
            receipt_path=run / "receipt.json",
            ledger_root=tmp_path / "ledger",
        )


def test_test_requires_consumed_receipt(tmp_path: Path) -> None:
    config, split = _write_contracts(tmp_path)
    run = tmp_path / "run"
    run.mkdir()
    with pytest.raises(TestAccessError, match="requires a consumed"):
        enforce_partition_access(
            "test",
            receipt_path=None,
            experiment_config=config,
            learning_split=split,
            run_root=run,
            ledger_root=tmp_path / "ledger",
        )
