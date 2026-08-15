from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(SCRIPTS / "common"))

from petct_models import (  # noqa: E402
    EDITOR_PRIMARY_ARCHITECTURE_ID,
    LEGAL_GOALS,
    OPERATION_TO_ID,
    P2T_ARCHITECTURE_IDS,
    P2T_PRIMARY_ARCHITECTURE_ID,
    TARGET_TO_ID,
    SCOPE_TO_ID,
    IntentEmbedding,
    ResidualEditorUNet2D,
    SimpleSignedCueIntentNet,
    build_p2t_model,
    decode_joint_goal,
    editor_condition_ids,
)


def _visual(batch: int = 2) -> torch.Tensor:
    visual = torch.randn(batch, 17, 32, 32)
    visual[:, 15:] = 0
    visual[0, 15, 12:16, 12:16] = 1
    if batch > 1:
        visual[1, 16, 18:22, 18:22] = 1
    return visual


def test_simple_first_p2t_emits_joint6_and_three_binary_auxiliary_heads() -> None:
    model = build_p2t_model(width=32)
    assert isinstance(model, SimpleSignedCueIntentNet)
    assert model.architecture_id == P2T_PRIMARY_ARCHITECTURE_ID
    assert P2T_ARCHITECTURE_IDS == ("simple_signed_scribble_state_pool_v2",)
    output = model(_visual(), torch.tensor([[1.0, 1.0], [2.0, 2.0]]))
    assert output["joint_logits"].shape == (2, 6)
    assert output["operation_logits"].shape == (2, 2)
    assert output["target_logits"].shape == (2, 2)
    assert output["scope_logits"].shape == (2, 2)
    assert "attention" not in output


def test_p2t_rejects_legacy_16_channel_and_invalid_mixed_polarity() -> None:
    model = build_p2t_model(width=32)
    with pytest.raises(ValueError, match="17 channels"):
        model(torch.zeros(1, 16, 32, 32), torch.ones(1, 2))
    visual = _visual(1)
    visual[0, 16, 2:4, 2:4] = 1
    with pytest.raises(ValueError, match="one polarity"):
        model(visual, torch.ones(1, 2))


def test_polarity_blind_arm_is_exact_duplicated_support_only() -> None:
    model = build_p2t_model(width=32)
    visual = _visual(1)
    visual[:, 16] = visual[:, 15]
    output = model(visual, torch.ones(1, 2))
    assert output["joint_logits"].shape == (1, 6)


def test_joint_decoder_is_the_only_final_slot_decoder() -> None:
    logits = torch.full((6, 6), -10.0)
    logits[torch.arange(6), torch.arange(6)] = 10.0
    joint, operation, target, scope = decode_joint_goal(logits)
    assert joint.tolist() == list(range(6))
    assert operation.tolist() == [0, 1, 0, 1, 0, 1]
    assert target.tolist() == [0, 0, 0, 0, 1, 1]
    assert scope.tolist() == [0, 0, 1, 1, 1, 1]
    assert LEGAL_GOALS == (
        "ADD_SAME_LOCAL",
        "REMOVE_SAME_LOCAL",
        "ADD_SAME_COMPLETE",
        "REMOVE_SAME_COMPLETE",
        "ADD_NEW_COMPLETE",
        "REMOVE_NEW_COMPLETE",
    )


def test_intent_embedding_allows_null_and_operation_only_but_rejects_illegal_tuple() -> None:
    embedding = IntentEmbedding(dim=16)
    null = embedding(
        torch.tensor([OPERATION_TO_ID["NULL"]]),
        torch.tensor([TARGET_TO_ID["NULL"]]),
        torch.tensor([SCOPE_TO_ID["NULL"]]),
    )
    assert torch.equal(null, torch.zeros_like(null))
    operation_only = embedding(
        torch.tensor([OPERATION_TO_ID["REMOVE"]]),
        torch.tensor([TARGET_TO_ID["NULL"]]),
        torch.tensor([SCOPE_TO_ID["NULL"]]),
    )
    assert operation_only.shape == (1, 16)
    with pytest.raises(ValueError, match="unsupported"):
        embedding(
            torch.tensor([OPERATION_TO_ID["ADD"]]),
            torch.tensor([TARGET_TO_ID["NEW"]]),
            torch.tensor([SCOPE_TO_ID["LOCAL"]]),
        )


def test_editor_condition_ids_fail_closed_on_retired_and_new_local() -> None:
    assert editor_condition_ids(
        "REMOVE", "SAME", "COMPLETE", "CORRECT"
    ) == (
        OPERATION_TO_ID["REMOVE"],
        TARGET_TO_ID["SAME"],
        SCOPE_TO_ID["COMPLETE"],
    )
    assert editor_condition_ids(
        "ADD", "SAME", "LOCAL", "OPERATION_ONLY"
    ) == (
        OPERATION_TO_ID["ADD"],
        TARGET_TO_ID["NULL"],
        SCOPE_TO_ID["NULL"],
    )
    with pytest.raises(ValueError, match="NEW_LOCAL"):
        editor_condition_ids("ADD", "NEW", "LOCAL", "CORRECT")
    for retired_target in ("ATTACHED", "STANDALONE"):
        with pytest.raises(ValueError, match="legal target"):
            editor_condition_ids("ADD", retired_target, "LOCAL", "CORRECT")


def test_simple_editor_applies_add_union_and_remove_subtraction() -> None:
    model = ResidualEditorUNet2D(base=8, intent_dim=16)
    assert model.architecture_id == EDITOR_PRIMARY_ARCHITECTURE_ID
    visual = torch.randn(2, 12, 16, 16)
    operation = torch.tensor([OPERATION_TO_ID["ADD"], OPERATION_TO_ID["REMOVE"]])
    target = torch.tensor([TARGET_TO_ID["SAME"], TARGET_TO_ID["NEW"]])
    scope = torch.tensor([SCOPE_TO_ID["LOCAL"], SCOPE_TO_ID["COMPLETE"]])
    logits = model(visual, operation, target, scope)
    m0 = torch.zeros_like(logits)
    m0[1, :, 4:8, 4:8] = 1
    delta, corrected = model.apply_operation(logits, m0, operation, threshold=0.0)
    assert corrected[0].all()
    assert not corrected[1].any()
    assert delta.shape == corrected.shape == logits.shape
