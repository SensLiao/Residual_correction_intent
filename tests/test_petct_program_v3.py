"""v3 program grammar / components / losses / model invariant tests.

Pure-python and small-tensor tests only; no server data or weights.
"""

import sys
import json
from pathlib import Path

import numpy as np
import pytest
import torch

SCRIPTS_ROOT = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from common.petct_components import (  # noqa: E402
    ENUMERATION_VERSION,
    cue_hit_component,
    enumerate_components,
)
from common.petct_program_contract import (  # noqa: E402
    ADD_FAMILIES,
    FAMILY_TO_GOAL,
    GOAL_TO_FAMILY,
    GRAMMAR_VERSION,
    NEW_CUE_SENTINEL,
    REMOVE_FAMILIES,
    ProgramContractError,
    compile_legal_call,
    family_ids,
    operation_from_cue_sign,
    rendered_slots,
    validate_legal_call,
)
from common.petct_program_learning import (  # noqa: E402
    GroupedBatchSampler,
    LearningContractError,
    load_label_manifest,
    matched_family_margin_loss,
    multi_positive_pointer_loss,
)
from common.petct_program_models import (  # noqa: E402
    ProgramCompilerNet,
    ProgramEditorUNet2D,
    ProgramEmbedding,
)


# ---------------------------------------------------------------- grammar


def test_operation_from_cue_sign():
    assert operation_from_cue_sign(True, False) == "ADD"
    assert operation_from_cue_sign(False, True) == "REMOVE"
    with pytest.raises(ProgramContractError):
        operation_from_cue_sign(True, True)
    with pytest.raises(ProgramContractError):
        operation_from_cue_sign(False, False)


def test_legal_families_per_operation():
    assert family_ids("ADD") == list(ADD_FAMILIES)
    assert family_ids("REMOVE") == list(REMOVE_FAMILIES)
    assert family_ids("REMOVE", include_repair=False) == ["TRIM_LOCAL", "DELETE_COMPONENT"]


def test_goal_family_roundtrip_covers_frozen_six():
    goals = {
        "ADD_SAME_LOCAL", "REMOVE_SAME_LOCAL", "ADD_SAME_COMPLETE",
        "REMOVE_SAME_COMPLETE", "ADD_NEW_COMPLETE", "REMOVE_NEW_COMPLETE",
    }
    assert set(FAMILY_TO_GOAL.values()) == goals
    assert set(GOAL_TO_FAMILY) == goals
    for goal in goals:
        assert FAMILY_TO_GOAL[GOAL_TO_FAMILY[goal]] == goal


def test_create_new_requires_new_cue_operand():
    with pytest.raises(ProgramContractError):
        validate_legal_call("ADD", "CREATE_NEW", "component_3")
    validate_legal_call("ADD", "CREATE_NEW", NEW_CUE_SENTINEL)
    with pytest.raises(ProgramContractError):
        validate_legal_call("REMOVE", "DELETE_COMPONENT", NEW_CUE_SENTINEL)
    with pytest.raises(ProgramContractError):
        validate_legal_call("ADD", "TRIM_LOCAL", "component_1")


def test_rendered_slots_are_deterministic_projections():
    assert rendered_slots("ADD", "GROW_LOCAL") == ("ADD", "SAME", "LOCAL")
    assert rendered_slots("ADD", "COMPLETE_EXISTING") == ("ADD", "SAME", "COMPLETE")
    assert rendered_slots("ADD", "CREATE_NEW") == ("ADD", "NEW", "COMPLETE")
    assert rendered_slots("REMOVE", "DELETE_COMPONENT") == ("REMOVE", "NEW", "COMPLETE")


def test_compile_legal_call_record():
    call = compile_legal_call("ADD", 1, "component_2")
    assert call["family"] == "COMPLETE_EXISTING"
    assert call["goal"] == "ADD_SAME_COMPLETE"
    assert call["grammar_version"] == GRAMMAR_VERSION


def test_program_schema_abstain_needs_no_fake_call_and_predict_mapping_is_typed():
    jsonschema = pytest.importorskip("jsonschema")
    schema_path = Path(__file__).resolve().parents[1] / "schemas" / "petct_program_v1.schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    validator = jsonschema.Draft202012Validator(schema)
    base = {
        "schema_version": "PETCT-PROGRAM-v1.0",
        "episode_id": "opaque-episode",
        "operation": "ADD",
        "confidence": None,
        "typed_trace": [],
        "audit": {
            "grammar_version": "PETCT-PROGRAM-GRAMMAR-v1.0",
            "enumeration_version": "v",
            "m_sha256": "a" * 64,
            "manifest_sha256": "b" * 64,
            "checkpoint_hash": None,
            "visibility_lane": "inference_visible",
        },
    }
    abstain = {**base, "decision": "ABSTAIN", "abstain_reason": "low confidence"}
    validator.validate(abstain)
    assert not validator.is_valid({**abstain, "family": "CREATE_NEW"})
    inconsistent = {
        **base,
        "decision": "PREDICT",
        "family": "CREATE_NEW",
        "operand": "NEW_CUE",
        "goal": "REMOVE_NEW_COMPLETE",
        "protected_refs": {
            "protected_complement": True,
            "monotone_update": True,
            "selected_component_scope": False,
            "new_cue_channel_all_zero": True,
        },
    }
    assert not validator.is_valid(inconsistent)


# -------------------------------------------------------------- components


def _two_boxes_volume():
    volume = np.zeros((8, 16, 16), dtype=np.uint8)
    volume[1:4, 2:7, 2:7] = 1  # box A
    volume[5:7, 10:14, 10:14] = 1  # box B, no face/edge contact -> separate
    return volume


def test_enumerate_components_two_boxes():
    volume = _two_boxes_volume()
    enumeration = enumerate_components(
        volume,
        episode_id="ep1",
        m_sha256="sha",
        spacing_xyz=np.array([3.0, 1.0, 1.0]),
    )
    assert len(enumeration.components) == 2
    assert enumeration.enumeration_version == ENUMERATION_VERSION
    assert enumeration.key().startswith("ep1|sha|")


def test_18_connectivity_splits_corner_only_contact():
    volume = np.zeros((5, 5, 5), dtype=np.uint8)
    volume[1, 1, 1] = 1
    volume[2, 2, 2] = 1  # corner-only neighbour of (1,1,1)
    enumeration = enumerate_components(
        volume,
        episode_id="ep2",
        m_sha256="sha",
        spacing_xyz=np.array([1.0, 1.0, 1.0]),
    )
    assert len(enumeration.components) == 2


def test_cue_hit_component_picks_max_overlap():
    volume = _two_boxes_volume()
    enumeration = enumerate_components(
        volume,
        episode_id="ep3",
        m_sha256="sha",
        spacing_xyz=np.array([3.0, 1.0, 1.0]),
        cue_voxels=np.array([[2, 4, 4]]),
    )
    assert cue_hit_component(enumeration, np.array([[2, 4, 4]]), volume) == 1
    assert cue_hit_component(enumeration, np.array([[6, 12, 12]]), volume) == 2
    assert cue_hit_component(enumeration, np.array([[4, 8, 8]]), volume) is None


# ------------------------------------------------------------------ losses


def test_margin_loss_penalizes_only_wrong_relative_preference():
    logits = torch.tensor([[3.0, 1.0, 0.5], [0.5, 3.0, 1.0]])
    targets = torch.tensor([0, 1])
    loss = matched_family_margin_loss(
        logits, targets, ["g1", "g1"], torch.tensor([0, 0]), margin=0.3
    )
    assert float(loss) == pytest.approx(0.0, abs=1e-6)
    # both rows prefer the sibling family: the pair margin must fire
    bad = torch.tensor([[1.0, 3.0, 0.5], [3.0, 0.5, 1.0]])
    loss_bad = matched_family_margin_loss(
        bad, targets, ["g1", "g1"], torch.tensor([0, 0]), margin=0.3
    )
    assert float(loss_bad) > 0.0


def test_margin_loss_rejects_mixed_operation_group():
    logits = torch.tensor([[3.0, 1.0, 0.5], [0.5, 1.0, 3.0]])
    targets = torch.tensor([0, 2])
    with pytest.raises(LearningContractError):
        matched_family_margin_loss(
            logits, targets, ["g1", "g1"], torch.tensor([0, 1]), margin=0.3
        )


def test_multi_positive_pointer_loss():
    logits = torch.tensor([[1.0, 2.0, 3.0]])
    mask = torch.tensor([[True, True, True]])
    loss_any = multi_positive_pointer_loss(logits, mask, [[0, 1, 2]])
    assert float(loss_any) == pytest.approx(0.0, abs=1e-6)
    loss_one = multi_positive_pointer_loss(logits, mask, [[0]])
    assert float(loss_one) > 0.0
    with pytest.raises(LearningContractError):
        multi_positive_pointer_loss(logits, mask, [[5]])


def test_grouped_batch_sampler_keeps_groups_whole():
    group_ids = ["a", "a", "a", "b", "b", "b", "c", "c", "c"]
    sampler = GroupedBatchSampler(group_ids, batch_size=4, seed=0)
    for batch in sampler:
        batch_groups = {group_ids[index] for index in batch}
        assert len(batch_groups) == 1


def test_label_manifest_rejects_six_rows_mixed_across_signed_operations(
    tmp_path: Path,
):
    path = tmp_path / "labels.jsonl"
    rows = []
    for index, goal in enumerate(
        (
            "ADD_SAME_LOCAL",
            "ADD_SAME_COMPLETE",
            "ADD_NEW_COMPLETE",
            "REMOVE_SAME_LOCAL",
            "REMOVE_SAME_COMPLETE",
            "REMOVE_NEW_COMPLETE",
        )
    ):
        operation = goal.split("_", 1)[0]
        rows.append(
            {
                "schema_version": "PETCT-PROGRAM-LABEL-MANIFEST-v1.0",
                "episode_id": "ep-%d" % index,
                "case_id": "case-a",
                "patient_id": "patient-a",
                "partition": "train",
                "goal": goal,
                "operation": operation,
                "matched_state_group_id": "incorrect-six-row-group",
                "evaluation_npz": "unused",
                "evaluation_sha256": "unused",
                "learning_split_sha256": "unused",
            }
        )
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    with pytest.raises(LearningContractError, match="same-operation triplet"):
        load_label_manifest(path)


# ------------------------------------------------------------------ models


def test_program_embedding_null_is_exact_zero():
    embedding = ProgramEmbedding(dim=16)
    output = embedding(
        torch.tensor([-1, -1, -1]),
        torch.tensor([2, 2, 2]),
        torch.tensor([1, 1, 1]),
    )
    assert torch.all(output == 0.0)


def test_compiler_forward_shapes_and_operation_disjoint():
    model = ProgramCompilerNet(width=32)
    model.eval()
    visual = torch.randn(2, 17, 32, 32)
    spacing = torch.ones(2, 2)
    operation_ids = torch.tensor([0, 1])
    vectors = torch.randn(2, 3, 7)
    mask = torch.tensor([[True, True, False], [True, False, False]])
    with torch.no_grad():
        output = model(visual, spacing, operation_ids, vectors, mask)
    assert output["family_logits"].shape == (2, 3)
    assert output["pointer_logits"].shape == (2, 3)
    # masked-out pointer candidates must carry -inf
    assert torch.isinf(output["pointer_logits"][0, 2])
    assert torch.isfinite(output["pointer_logits"][0, 0])


def test_editor_add_algebra_is_monotone_and_remove_is_restricted():
    logits = torch.tensor([[[[10.0, -10.0], [-10.0, 10.0]]]])
    m0 = torch.tensor([[[[1.0, 0.0], [0.0, 0.0]]]])
    selected = torch.zeros_like(m0)
    delta_add, corrected_add = ProgramEditorUNet2D.apply_program_operation(
        logits, m0, selected, torch.tensor([0]), threshold=0.5
    )
    # ADD: predicted new voxels only enter where M0 was empty; M0 voxel kept.
    assert corrected_add[0, 0, 0, 0] == 1.0
    assert corrected_add[0, 0, 1, 1] == 1.0
    assert corrected_add[0, 0, 0, 1] == 0.0
    selected_remove = torch.tensor([[[[0.0, 0.0], [1.0, 1.0]]]])
    m0_remove = torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]]])
    logits_remove = torch.tensor([[[[-10.0, -10.0], [10.0, 10.0]]]])
    delta_remove, corrected_remove = ProgramEditorUNet2D.apply_program_operation(
        logits_remove, m0_remove, selected_remove, torch.tensor([1]), threshold=0.5
    )
    # REMOVE: only the selected-component voxel at (1,0) can be deleted.
    assert corrected_remove[0, 0, 1, 0] == 0.0
    assert corrected_remove[0, 0, 0, 0] == 1.0  # protected


def test_editor_accepts_selected_channel_for_add_existing_without_deleting_m0():
    logits = torch.zeros(1, 1, 2, 2)
    m0 = torch.tensor([[[[1.0, 0.0], [0.0, 0.0]]]])
    selected = torch.ones(1, 1, 2, 2)
    _, corrected = ProgramEditorUNet2D.apply_program_operation(
        logits, m0, selected, torch.tensor([0]), threshold=0.5
    )
    assert corrected[0, 0, 0, 0]
