from __future__ import annotations

import sys
from pathlib import Path

import pytest


PROJECT = Path(__file__).resolve().parents[1]
SCRIPTS = PROJECT / "scripts"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(SCRIPTS / "common"))

import petct_learning  # noqa: E402
from petct_models import (  # noqa: E402
    P2T_PRIMARY_ARCHITECTURE_ID,
    validate_p2t_architecture_selection,
)
from petct_learning import (  # noqa: E402
    LearningContractError,
    load_editor_fusion_contract,
    load_experiment_config,
    load_intent_intervention_contract,
    load_p2t_evaluation_contract,
    load_training_contract,
    select_frozen_seed,
    validate_frozen_override,
    write_bytes_bundle_exclusive,
)


def test_frozen_config_drives_training_evaluation_and_intervention_contracts() -> None:
    config = load_experiment_config(
        PROJECT / "configs" / "petct_route_a_experiment.json"
    )
    p2t = load_training_contract(config, "p2t")
    editor = load_training_contract(config, "editor")
    evaluation = load_p2t_evaluation_contract(config)
    intervention = load_intent_intervention_contract(config)
    fusion = load_editor_fusion_contract(config)

    assert p2t["epochs"] == 100
    assert p2t["seeds"] == [3407, 4269, 9281]
    assert config["p2t"]["primary_architecture_id"] == P2T_PRIMARY_ARCHITECTURE_ID
    assert (
        validate_p2t_architecture_selection(
            config, P2T_PRIMARY_ARCHITECTURE_ID, "full"
        )
        == "primary"
    )
    assert editor["epochs"] == 150
    assert evaluation["primary_end_to_end_arm"] == "full"
    assert evaluation["ablation_inputs"] == [
        "full", "no_M0", "polarity_blind", "geometry_only"
    ]
    assert intervention == {
        "shuffle_algorithm": "six-legal-joint-label-preserving-non-identity-derangement-v2",
        "shuffle_seed": 20260717,
        "null_semantics": "zero-condition-vector-and-explicit-conditioner-bypass",
        "operation_only_semantics": "preserve-gold-operation-and-neutralize-target-scope",
        "wrong_scope_semantics": (
            "flip LOCAL and COMPLETE only when target=SAME; retain target; "
            "mark all NEW episodes INELIGIBLE"
        ),
            "wrong_operation_semantics": (
                "flip only the ADD/REMOVE conditioning token; retain the gold "
                "execution operation for physical set algebra and scoring; never "
                "construct a gold target or confirmatory utility row"
            ),
    }
    assert fusion == {
        "primary_architecture_id": "simple_operation_conditioned_residual_unet_v2",
        "deferred_ablations": [
            "FiLM", "concat", "gated", "intent-image cross_attention"
        ],
        "fusion_plan": "simple-first-current-campaign; listed ablations deferred",
    }


def test_seed_and_legacy_cli_values_can_only_assert_frozen_config() -> None:
    contract = {"seeds": [7, 11]}
    assert select_frozen_seed(contract, 7) == 7
    with pytest.raises(LearningContractError, match="frozen seed registry"):
        select_frozen_seed(contract, 13)
    validate_frozen_override("--epochs", 100, 100)
    with pytest.raises(LearningContractError, match="differs from frozen"):
        validate_frozen_override("--epochs", 99, 100)


def test_atomic_multi_file_publish_rolls_back_if_one_commit_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = tmp_path / "predictions.jsonl"
    second = tmp_path / "metrics.json"
    real_link = petct_learning.os.link
    calls = 0

    def fail_second_link(source: str, target: str) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("simulated second-file commit failure")
        real_link(source, target)

    monkeypatch.setattr(petct_learning.os, "link", fail_second_link)
    with pytest.raises(OSError, match="simulated"):
        write_bytes_bundle_exclusive({first: b"one\n", second: b"two\n"})
    assert not first.exists()
    assert not second.exists()
    assert not list(tmp_path.glob(".*.tmp"))
