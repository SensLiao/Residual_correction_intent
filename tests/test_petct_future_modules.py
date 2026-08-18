"""Contract tests for the deferred architecture ladder (inactive by design).

Every architecture here must stay OUT of the active experiment configs until a
separate preregistration lands; these tests pin the shape/identity contracts
and the inactivity boundary, not any scientific claim.
"""

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "scripts"))

from common.petct_future_modules import (  # noqa: E402
    _ConvNormAct3D,
    _channel_norm,
    FUSION_DUAL_STEM_ARCHITECTURE_ID,
    FUSION_MODALITY_GATE_ARCHITECTURE_ID,
    FUSION_PET_CT_CROSS_ATTN_ARCHITECTURE_ID,
    EDITOR_FILM_ARCHITECTURE_ID,
    EDITOR_CONCAT_ARCHITECTURE_ID,
    EDITOR_GATED_ARCHITECTURE_ID,
    EDITOR_INTENT_CROSS_ATTN_ARCHITECTURE_ID,
    EDITOR_3D_ARCHITECTURE_ID,
    EDITOR_RES_ARCHITECTURE_ID,
    COMPILER_3D_ARCHITECTURE_ID,
    DualShallowStemBottleneckFusion,
    IntentImageCrossAttention,
    MisalignmentDA,
    ModalityGatedFusion,
    PETCTBidirectionalCrossAttention,
    ProgramCompilerNet3D,
    ProgramConcatConditioning,
    ProgramEditorUNet2DRes,
    ProgramEditorUNet3D,
    ProgramFiLMConditioning,
    ProgramGatedConditioning,
    apply_program_operation_3d,
)
from common.petct_models import (  # noqa: E402
    P2T_ARCHITECTURE_IDS,
    P2T_DEFERRED_CROSS_ATTENTION_ARCHITECTURE_ID,
)
from common.petct_program_models import (  # noqa: E402
    PROGRAM_EDITOR_ARCHITECTURE_ID,
    NULL_FAMILY_ID,
)

ALL_DEFERRED_IDS = [
    EDITOR_FILM_ARCHITECTURE_ID,
    EDITOR_CONCAT_ARCHITECTURE_ID,
    EDITOR_GATED_ARCHITECTURE_ID,
    EDITOR_INTENT_CROSS_ATTN_ARCHITECTURE_ID,
    FUSION_DUAL_STEM_ARCHITECTURE_ID,
    FUSION_MODALITY_GATE_ARCHITECTURE_ID,
    FUSION_PET_CT_CROSS_ATTN_ARCHITECTURE_ID,
    EDITOR_3D_ARCHITECTURE_ID,
    EDITOR_RES_ARCHITECTURE_ID,
    COMPILER_3D_ARCHITECTURE_ID,
]


def test_deferred_architecture_ids_stay_out_of_active_sets() -> None:
    """The inactivity boundary: no deferred id may enter the active P2T set or
    collide with the frozen active editor id."""
    assert all(arch_id not in P2T_ARCHITECTURE_IDS for arch_id in ALL_DEFERRED_IDS)
    assert P2T_DEFERRED_CROSS_ATTENTION_ARCHITECTURE_ID not in P2T_ARCHITECTURE_IDS
    assert all(arch_id != PROGRAM_EDITOR_ARCHITECTURE_ID for arch_id in ALL_DEFERRED_IDS)
    assert len(set(ALL_DEFERRED_IDS)) == len(ALL_DEFERRED_IDS)


@pytest.fixture()
def visual_13() -> torch.Tensor:
    return torch.randn(2, 13, 32, 32)


@pytest.fixture()
def program_inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    family_ids_t = torch.tensor([0, NULL_FAMILY_ID])
    operand_mode = torch.tensor([0, 2])
    support_mode = torch.tensor([0, 1])
    return family_ids_t, operand_mode, support_mode


def test_film_conditioning_null_bypass_and_shape() -> None:
    module = ProgramFiLMConditioning(program_dim=64, channels=64, num_scales=2)
    features = [torch.randn(2, 64, 16, 16), torch.randn(2, 64, 8, 8)]
    condition = torch.randn(2, 64)
    bypass = torch.tensor([False, True])
    outputs = module(features, condition, bypass)
    assert len(outputs) == 2
    assert all(out.shape == feature.shape for out, feature in zip(outputs, features))
    assert torch.allclose(outputs[0][1], features[0][1])
    assert torch.allclose(outputs[1][1], features[1][1])
    assert module.parameter_count() > 0


def test_concat_conditioning_null_bypass_and_shape() -> None:
    module = ProgramConcatConditioning(program_dim=64, bottleneck_channels=64)
    bottleneck = torch.randn(2, 64, 8, 8)
    condition = torch.randn(2, 64)
    bypass = torch.tensor([False, True])
    out = module(bottleneck, condition, bypass)
    assert out.shape == bottleneck.shape
    assert torch.allclose(out[1], bottleneck[1])
    assert module.parameter_count() > 0


def test_gated_conditioning_null_bypass_and_shape() -> None:
    module = ProgramGatedConditioning(program_dim=64, channels=64)
    feature = torch.randn(2, 64, 8, 8)
    condition = torch.randn(2, 64)
    bypass = torch.tensor([False, True])
    out = module(feature, condition, bypass)
    assert out.shape == feature.shape
    assert torch.allclose(out[1], feature[1])
    assert module.parameter_count() > 0


def test_intent_image_cross_attention_shape() -> None:
    module = IntentImageCrossAttention(program_dim=64, channels=64, heads=4)
    feature = torch.randn(2, 64, 8, 8)
    condition = torch.randn(2, 64)
    bypass = torch.tensor([False, True])
    out = module(feature, condition, bypass)
    assert out.shape == feature.shape
    assert torch.allclose(out[1], feature[1])
    assert module.parameter_count() > 0


def test_dual_stem_fusion_shape(visual_13: torch.Tensor) -> None:
    module = DualShallowStemBottleneckFusion(base=16)
    out = module(visual_13)
    assert out.shape == (2, 32, 32, 32)


def test_modality_gate_fusion_shape() -> None:
    module = ModalityGatedFusion(base=16)
    pet, ct = torch.randn(2, 5, 32, 32), torch.randn(2, 5, 32, 32)
    pet_map, ct_map = module(pet, ct)
    assert pet_map.shape == (2, 16, 32, 32)
    assert ct_map.shape == (2, 16, 32, 32)


def test_petct_cross_attention_shape() -> None:
    module = PETCTBidirectionalCrossAttention(channels=32, heads=4)
    pet_map, ct_map = torch.randn(2, 32, 16, 16), torch.randn(2, 32, 16, 16)
    out = module(pet_map, ct_map)
    assert out.shape == (2, 32, 16, 16)


def test_misalignment_da_only_moves_selected_modality(visual_13: torch.Tensor) -> None:
    module = MisalignmentDA(max_shift=2)
    shifts = torch.tensor([[1, 1], [0, 0]])
    out = module(visual_13, shifts, modality="ct")
    # PET block untouched
    assert torch.allclose(out[:, :5], visual_13[:, :5])
    # CT block of batch 0 shifted; batch 1 with (0,0) unchanged
    assert torch.allclose(out[1, 5:10], visual_13[1, 5:10])
    assert not torch.allclose(out[0, 5:10], visual_13[0, 5:10])
    # context channels untouched
    assert torch.allclose(out[:, 10:], visual_13[:, 10:])
    with pytest.raises(ValueError):
        module(visual_13, torch.tensor([[3, 0], [0, 0]]), modality="pet")


def test_editor_3d_program_shape_and_null_bypass(
    visual_13: torch.Tensor, program_inputs: tuple
) -> None:
    family_ids_t, operand_mode, support_mode = program_inputs
    module = ProgramEditorUNet3D(base=8)
    volume = torch.randn(2, 13, 16, 16, 16)
    out = module(volume, family_ids_t, operand_mode, support_mode)
    assert out.shape == (2, 1, 16, 16, 16)
    assert module.parameter_count() > 0
    # NULL row must produce the exact neutral conditioning shift path
    null_out = module(volume[1:2], family_ids_t[1:2], operand_mode[1:2], support_mode[1:2])
    assert torch.isfinite(null_out).all()


def test_editor_3d_continuous_requires_state_and_mask() -> None:
    module = ProgramEditorUNet3D(base=8, conditioner="continuous")
    volume = torch.randn(2, 13, 16, 16, 16)
    with pytest.raises(ValueError):
        module(volume, torch.tensor([0, 0]), torch.tensor([0, 0]), torch.tensor([0, 0]))
    state = torch.randn(2, 64)
    active = torch.tensor([True, False])
    out = module(
        volume,
        torch.tensor([0, 0]),
        torch.tensor([0, 0]),
        torch.tensor([0, 0]),
        state_embedding=state,
        active_mask=active,
    )
    assert out.shape == (2, 1, 16, 16, 16)


def test_apply_program_operation_3d_algebra() -> None:
    m0 = torch.zeros(1, 1, 8, 8, 8, dtype=torch.float32)
    m0[0, 0, 2:6, 2:6, 2:6] = 1.0
    component = m0.clone()
    logits = torch.full((1, 1, 8, 8, 8), -10.0)
    logits[0, 0, 5:7, 5:7, 5:7] = 10.0  # outside current mask
    delta, corrected = apply_program_operation_3d(
        logits, m0, component, torch.tensor([0])
    )
    # ADD is monotone union
    assert torch.all(corrected >= m0)
    assert corrected.sum() > m0.sum()
    # REMOVE is complement-protected: prediction outside the component cannot
    # delete anything, and no new voxels may appear.
    logits_rm = torch.full((1, 1, 8, 8, 8), -10.0)
    logits_rm[0, 0, 6:8, 6:8, 6:8] = 10.0  # fully outside the component (2:6)
    delta_rm, corrected_rm = apply_program_operation_3d(
        logits_rm, m0, component, torch.tensor([1])
    )
    assert torch.all(corrected_rm <= m0)
    assert corrected_rm.sum() == m0.sum()  # prediction was outside the component
    # REMOVE inside the component DOES delete (complement protection passes).
    logits_in = torch.full((1, 1, 8, 8, 8), -10.0)
    logits_in[0, 0, 2:3, 2:6, 2:6] = 10.0
    _, corrected_in = apply_program_operation_3d(
        logits_in, m0, component, torch.tensor([1])
    )
    assert corrected_in.sum() < m0.sum()
    with pytest.raises(ValueError):
        apply_program_operation_3d(logits, m0, component, torch.tensor([2]))


def test_editor_res_backbone_shape_and_count(
    visual_13: torch.Tensor, program_inputs: tuple
) -> None:
    family_ids_t, operand_mode, support_mode = program_inputs
    module = ProgramEditorUNet2DRes(base=16)
    out = module(visual_13, family_ids_t, operand_mode, support_mode)
    assert out.shape == (2, 1, 32, 32)
    assert module.parameter_count() > 0


def test_compiler_3d_shape_and_masking() -> None:
    module = ProgramCompilerNet3D(state_channels=15, width=16)
    visual = torch.randn(2, 17, 16, 16, 16)
    operation_ids = torch.tensor([0, 1])
    component_vectors = torch.randn(2, 3, 7)
    component_mask = torch.tensor([[True, True, False], [True, False, False]])
    out = module(visual, operation_ids, component_vectors, component_mask)
    assert out["family_logits"].shape == (2, 3)
    assert out["pointer_logits"].shape == (2, 3)
    assert out["embedding"].shape == (2, 16)
    # masked candidates get -inf pointers
    assert torch.isneginf(out["pointer_logits"][0, 2])
    assert torch.isneginf(out["pointer_logits"][1, 1])
    assert module.parameter_count() > 0


def test_future_modules_deterministic_under_seed() -> None:
    torch.manual_seed(3407)
    module_a = ProgramEditorUNet2DRes(base=16)
    weight_a = module_a.output.weight.detach().clone()
    torch.manual_seed(3407)
    module_b = ProgramEditorUNet2DRes(base=16)
    assert torch.equal(weight_a, module_b.output.weight.detach())


# --------------------------------------------------------------------------
# 2026-08-18 external-review fixes (RED-first regressions)
# --------------------------------------------------------------------------


def test_channel_norm_accepts_non_divisible_channels() -> None:
    for channels in (12, 13, 24, 48):
        out = _channel_norm(channels)(torch.zeros(2, channels, 8, 8))
        assert out.shape == (2, channels, 8, 8)


def test_conv_norm_act_3d_accepts_12_channels() -> None:
    block = _ConvNormAct3D(5, 12)
    assert block(torch.zeros(1, 5, 8, 8, 8)).shape == (1, 12, 8, 8, 8)


def test_film_per_scale_channels() -> None:
    module = ProgramFiLMConditioning(
        program_dim=16, channels_per_scale=(4, 8, 16)
    )
    features = [torch.randn(2, c, 8, 8) for c in (4, 8, 16)]
    outputs = module(features, torch.randn(2, 16), torch.zeros(2))
    assert [out.shape[1] for out in outputs] == [4, 8, 16]
    # length mismatch still fails fast
    with pytest.raises(ValueError):
        module(features[:2], torch.randn(2, 16), torch.zeros(2))
    # wrong per-scale width is caught explicitly
    with pytest.raises(ValueError):
        module([torch.randn(2, 4, 8, 8)] * 3, torch.randn(2, 16), torch.zeros(2))


def test_compiler_3d_pointer_all_masked_is_finite() -> None:
    module = ProgramCompilerNet3D(state_channels=15, width=16)
    visual = torch.randn(1, 17, 16, 16, 16)
    out = module(
        visual,
        torch.tensor([0]),
        torch.randn(1, 3, 7),
        torch.zeros(1, 3, dtype=torch.bool),
    )
    probs = out["pointer_logits"].softmax(dim=-1)
    assert torch.isfinite(probs).all()
    assert torch.allclose(probs.sum(dim=-1), torch.ones(1))


def test_compiler_3d_pointer_mask_respected() -> None:
    module = ProgramCompilerNet3D(state_channels=15, width=16)
    visual = torch.randn(1, 17, 16, 16, 16)
    mask = torch.tensor([[True, False, True]])
    out = module(visual, torch.tensor([0]), torch.randn(1, 3, 7), mask)
    probs = out["pointer_logits"].softmax(dim=-1)
    assert probs[0, 1].item() == 0.0
    assert torch.isfinite(probs).all()


def test_misalignment_da_no_wraparound() -> None:
    module = MisalignmentDA(max_shift=2)
    visual = torch.zeros(1, 13, 4, 6)
    visual[0, 5, 1, 0] = 1.0  # far-left marker in the CT block
    shifted = module(visual, torch.tensor([[0, 1]]), modality="ct")
    assert shifted[0, 5, 1, 1].item() == 1.0  # marker moved right by one
    assert shifted[0, 5, 1, -1].item() == 0.0  # nothing wrapped around
    assert shifted[0, 5, 1, 0].item() == 0.0  # vacated column zero-filled
    # right-edge marker must be DROPPED under +1 shift, never wrapped to col 0
    edge = torch.zeros(1, 13, 4, 6)
    edge[0, 5, 1, -1] = 1.0
    shifted_edge = module(edge, torch.tensor([[0, 1]]), modality="ct")
    assert shifted_edge[0, 5, 1, 0].item() == 0.0
    assert shifted_edge.sum().item() == 0.0
    # (0, 0) shift is identity
    same = module(visual, torch.tensor([[0, 0]]), modality="ct")
    assert torch.allclose(same, visual)


def test_petct_cross_attention_resolution_guard() -> None:
    module = PETCTBidirectionalCrossAttention(channels=32, heads=4)
    with pytest.raises(ValueError, match="pooled|windowed"):
        module(torch.randn(2, 32, 65, 65), torch.randn(2, 32, 65, 65))


def test_modality_gate_evaluates_stem_once() -> None:
    module = ModalityGatedFusion(base=16)
    calls: dict[str, int] = {"pet": 0, "ct": 0}

    class CountingStem(nn.Module):
        def __init__(self, inner: nn.Module, key: str):
            super().__init__()
            self.inner = inner
            self.key = key

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            calls[self.key] += 1
            return self.inner(x)

    module.pet_stem = CountingStem(module.pet_stem, "pet")
    module.ct_stem = CountingStem(module.ct_stem, "ct")
    pet_map, ct_map = module(
        torch.randn(2, 5, 32, 32), torch.randn(2, 5, 32, 32)
    )
    assert calls == {"pet": 1, "ct": 1}
    assert pet_map.shape == (2, 16, 32, 32)
    assert ct_map.shape == (2, 16, 32, 32)


def test_dual_stem_exposes_visual_channels() -> None:
    module = DualShallowStemBottleneckFusion(base=16)
    assert module.visual_channels == 13
    with pytest.raises(ValueError):
        DualShallowStemBottleneckFusion(visual_channels=11)
