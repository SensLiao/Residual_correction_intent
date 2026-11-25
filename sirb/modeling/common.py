"""Building blocks and helpers shared by the SIRB modules."""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import nullcontext

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from ..data.geometry import feature_affine_from_conv, grid_world_points, sample_world_features
from ..data.observations import STATE_CHANNELS, VOXEL_CHANNELS, WHOLE_VOLUME_CHANNELS
from ..data.roles import CORRECT_TISSUE, ROLE_COUNT

__all__ = ["FIXED_CORRECT_TISSUE_LOGIT", "KEPT_WITHOUT_SCRIBBLE", "DepthwiseConvBlock3d", "EncoderStage",
           "ResidualBlock3d", "conv_feature_affine", "fix_correct_tissue_logit", "fp32_region", "run_checkpointed",
           "sample_to_grid", "validate_widths", "zero_channels"]

#: The correct-tissue logit wherever role logits are produced when SIRB is built without the correct-tissue role
#: (the No correct-tissue role variant). It is finite, so every finiteness check holds; exp(-30) is about 9e-14.
FIXED_CORRECT_TISSUE_LOGIT = -30.0

#: The input channels that SIRB still reads without scribble conditioning (the No scribble conditioning variant):
#: the images, the mask and its covered fraction, and for native voxels also their offsets and spacing. Every other
#: channel is replaced by zero.
_KEPT_NAMES = {"whole_volume": ("pet", "ct", "mask", "covered_fraction"),
               "state": ("mask", "covered_fraction"),
               "voxel": ("pet", "ct", "mask", "offset_x", "offset_y", "offset_z", "spacing_x_mm", "spacing_y_mm",
                         "spacing_z_mm")}
KEPT_WITHOUT_SCRIBBLE = {
    lane: tuple(name in _KEPT_NAMES[lane] for name in channels)
    for lane, channels in (("whole_volume", WHOLE_VOLUME_CHANNELS), ("state", STATE_CHANNELS),
                           ("voxel", VOXEL_CHANNELS))}


def validate_widths(widths: Sequence[int], name: str) -> tuple[int, ...]:
    result = tuple(int(width) for width in widths)
    if not result or any(width <= 0 or width % 8 for width in result):
        raise ValueError(f"{name} must contain positive widths divisible by eight (group normalisation); "
                         f"got {result}")
    return result


class ResidualBlock3d(nn.Module):
    """Two 3^3 convolutions with group normalisation (8 groups) and LeakyReLU(0.01), and a shortcut that is a 1^3
    convolution when the width changes."""

    def __init__(self, cin: int, cout: int) -> None:
        super().__init__()
        validate_widths((cout,), "residual output")
        self.conv1 = nn.Conv3d(cin, cout, 3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(8, cout, affine=True)
        self.conv2 = nn.Conv3d(cout, cout, 3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(8, cout, affine=True)
        self.shortcut = nn.Identity() if cin == cout else nn.Conv3d(cin, cout, 1, bias=False)
        self.activation = nn.LeakyReLU(0.01, inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.activation(self.norm1(self.conv1(x)))
        return self.activation(self.norm2(self.conv2(residual)) + self.shortcut(x))


class DepthwiseConvBlock3d(nn.Module):
    """Depthwise 3^3 convolution, pointwise 1^3 convolution, group normalisation (8 groups), GELU."""

    def __init__(self, cin: int, cout: int, *, stride: int = 1) -> None:
        super().__init__()
        validate_widths((cout,), "depthwise block output")
        self.depthwise = nn.Conv3d(cin, cin, 3, stride=stride, padding=1, groups=cin, bias=False)
        self.pointwise = nn.Conv3d(cin, cout, 1, bias=False)
        self.norm = nn.GroupNorm(8, cout, affine=True)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.norm(self.pointwise(self.depthwise(x))))


class EncoderStage(nn.Module):
    """One encoder level: a 3^3 convolution (stride 1 or 2) followed by residual blocks."""

    def __init__(self, conv: nn.Conv3d, blocks: Sequence[nn.Module]) -> None:
        super().__init__()
        self.conv = conv
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        for block in self.blocks:
            x = block(x)
        return x


def run_checkpointed(owner: nn.Module, module: nn.Module, *inputs: torch.Tensor) -> torch.Tensor:
    """``module(*inputs)``, recomputed in the backward pass when ``owner.activation_checkpointing`` is set while
    training with gradients (memory only; the values are the same)."""
    if owner.activation_checkpointing and owner.training and torch.is_grad_enabled():
        return checkpoint(module, *inputs, use_reentrant=False)
    return module(*inputs)


def conv_feature_affine(affine_xyz: torch.Tensor, convolution: nn.Conv3d) -> torch.Tensor:
    """Affine of a convolution's output grid, read from the layer's own kernel, stride, padding and dilation."""
    return feature_affine_from_conv(affine_xyz, kernel_size=convolution.kernel_size, stride=convolution.stride,
                                    padding=convolution.padding, dilation=convolution.dilation)


def sample_to_grid(features: torch.Tensor, feature_affine_xyz: torch.Tensor, target_shape_zyx: tuple[int, ...],
                   target_affine_xyz: torch.Tensor, *, support_affine_xyz: torch.Tensor,
                   support_shape_zyx: tuple[int, ...], support_valid_zyx: torch.Tensor | None = None
                   ) -> tuple[torch.Tensor, torch.Tensor]:
    """Trilinear samples of ``features`` at the cell centres of a target grid, as BCZYX, and their validity."""
    points = grid_world_points(target_shape_zyx, target_affine_xyz, device=features.device, dtype=torch.float64)
    points = points.reshape(-1, 3) if points.ndim == 4 else points.reshape(points.shape[0], -1, 3)
    sampled, valid = sample_world_features(features, feature_affine_xyz, points, support_affine_xyz=support_affine_xyz,
                                           support_shape_zyx=support_shape_zyx, support_valid_zyx=support_valid_zyx)
    return sampled.transpose(1, 2).reshape(features.shape[0], features.shape[1], *target_shape_zyx), valid


def fp32_region(tensor: torch.Tensor):
    """Autocast disabled (role arithmetic stays FP32); meta tensors have no autocast."""
    if tensor.device.type == "meta":
        return nullcontext()
    return torch.autocast(device_type=tensor.device.type, enabled=False)


def fix_correct_tissue_logit(logits: torch.Tensor, dim: int, value: float = FIXED_CORRECT_TISSUE_LOGIT) -> torch.Tensor:
    """Role logits whose correct-tissue entry along ``dim`` is the constant ``value``; the others pass unchanged."""
    if logits.shape[dim] != ROLE_COUNT:
        raise ValueError(f"role axis {dim} actual={logits.shape[dim]}, expected={ROLE_COUNT}")
    before = logits.narrow(dim, 0, CORRECT_TISSUE)
    constant = torch.full_like(logits.narrow(dim, CORRECT_TISSUE, 1), value)
    after = logits.narrow(dim, CORRECT_TISSUE + 1, ROLE_COUNT - CORRECT_TISSUE - 1)
    return torch.cat((before, constant, after), dim)


def zero_channels(tensor: torch.Tensor, dim: int, kept: Sequence[bool]) -> torch.Tensor:
    """``tensor`` with every channel of axis ``dim`` whose ``kept`` entry is False replaced by +0.0."""
    if tensor.shape[dim] != len(kept):
        raise ValueError(f"channel axis {dim} actual={tensor.shape[dim]}, expected={len(kept)}")
    pieces, start = [], 0
    for index in range(1, len(kept) + 1):
        if index == len(kept) or kept[index] != kept[start]:
            part = tensor.narrow(dim, start, index - start)
            pieces.append(part if kept[start] else torch.zeros_like(part))
            start = index
    return torch.cat(pieces, dim)
