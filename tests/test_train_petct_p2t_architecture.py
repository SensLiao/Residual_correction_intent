"""Static launcher/contract tests; no training or inference is executed."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "scripts"))

from common.petct_learning import (  # noqa: E402
    load_p2t_evaluation_contract,
    load_training_contract,
)
from common.petct_models import (  # noqa: E402
    P2T_ARCHITECTURE_IDS,
    P2T_DEFERRED_CROSS_ATTENTION_ARCHITECTURE_ID,
    P2T_PRIMARY_ARCHITECTURE_ID,
    build_p2t_model,
    p2t_architecture_contract,
)


def _config() -> dict:
    return json.loads(
        (PROJECT / "configs" / "petct_route_a_experiment.json").read_text(
            encoding="utf-8"
        )
    )


def test_simple_first_is_the_only_current_launcher_architecture() -> None:
    assert P2T_ARCHITECTURE_IDS == (P2T_PRIMARY_ARCHITECTURE_ID,)
    model = build_p2t_model()
    assert model.architecture_id == P2T_PRIMARY_ARCHITECTURE_ID
    assert p2t_architecture_contract(
        P2T_PRIMARY_ARCHITECTURE_ID, use_relative_geometry=True
    )["cross_attention"] is False
    with pytest.raises(ValueError, match="unsupported"):
        build_p2t_model(P2T_DEFERRED_CROSS_ATTENTION_ARCHITECTURE_ID)


def test_p2t_training_contract_has_joint_and_three_auxiliary_losses() -> None:
    contract = load_training_contract(_config(), "p2t")
    assert contract["joint_loss_weight"] == 1.0
    assert [
        contract["operation_loss_weight"],
        contract["target_loss_weight"],
        contract["scope_loss_weight"],
    ] == [0.25, 0.25, 0.25]


def test_confirmatory_execution_remains_fail_closed_pre_freeze() -> None:
    evaluation = load_p2t_evaluation_contract(_config())
    assert evaluation["validation_role"] == "DESCRIPTIVE_FEASIBILITY_ONLY"
    assert evaluation["confirmatory_execution_allowed"] is False
    assert evaluation["confirmatory_execution_gate"].startswith("BLOCKED_UNTIL")
