"""The whole-volume encoder: all 14 input channels of the whole examination at low resolution.

It reads the whole-volume grid (96 x 64 x 64 cells over the examination's field of view) and gives context beyond
the local window: its features enter the role decoder at the 12 mm level and are pooled into the source vectors.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from .common import EncoderStage, ResidualBlock3d, conv_feature_affine, run_checkpointed, validate_widths

__all__ = ["WholeVolumeEncoder"]


class WholeVolumeEncoder(nn.Module):
    """Three levels, each a stride-2 3^3 convolution and one residual block (widths 16, 32, 64)."""

    def __init__(self, widths: Sequence[int] = (16, 32, 64), *, in_channels: int = 14) -> None:
        super().__init__()
        self.widths = validate_widths(widths, "whole-volume widths")
        self.in_channels = int(in_channels)
        self.activation_checkpointing = False
        self.stages = nn.ModuleList()
        previous = self.in_channels
        for width in self.widths:
            self.stages.append(EncoderStage(nn.Conv3d(previous, width, 3, stride=2, padding=1, bias=False),
                                            [ResidualBlock3d(width, width)]))
            previous = width

    def forward(self, whole_volume_input: torch.Tensor) -> torch.Tensor:
        if whole_volume_input.ndim != 5 or whole_volume_input.shape[1] != self.in_channels:
            raise ValueError(f"expected whole-volume input [B,{self.in_channels},Z,Y,X]; "
                             f"got {tuple(whole_volume_input.shape)}")
        x = whole_volume_input
        for stage in self.stages:
            x = run_checkpointed(self, stage, x)
        return x

    def feature_affine(self, input_affine_xyz: torch.Tensor) -> torch.Tensor:
        """Affine of the output grid, from the actual strides of the stage convolutions."""
        frame = input_affine_xyz
        for stage in self.stages:
            frame = conv_feature_affine(frame, stage.conv)
        return frame
