"""The local encoder: an image stream on PET and CT fused with a state stream on the 12 state channels.

It reads one window of the 3 mm grid and has four levels (cells of 3, 6, 12 and 24 mm). At every level a
:class:`StateFusion3d` lets the state features scale and shift the image features (feature-wise linear modulation)
behind a learned scalar gate; the scale starts at zero, so the fusion begins as a gated addition. Each downsampling
is a stride-2, kernel-3, padding-1 convolution, so feature cell 0 stays on input cell 0
(:meth:`LocalEncoder.feature_affines`).
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from .common import (DepthwiseConvBlock3d, EncoderStage, ResidualBlock3d, conv_feature_affine, run_checkpointed,
                     validate_widths)

__all__ = ["LocalEncoder", "StateFusion3d"]

#: Initial value of the fusion gate.
INITIAL_GATE = 0.1


class StateFusion3d(nn.Module):
    """``image + gate * (project(state) + scale(state) * GN(image) + shift(state))``, with 1^3 convolutions."""

    def __init__(self, image_channels: int, state_channels: int) -> None:
        super().__init__()
        self.project = nn.Conv3d(state_channels, image_channels, 1, bias=False)
        self.scale = nn.Conv3d(state_channels, image_channels, 1, bias=False)
        self.shift = nn.Conv3d(state_channels, image_channels, 1, bias=False)
        self.norm = nn.GroupNorm(8, image_channels, affine=True)
        self.gate = nn.Parameter(torch.tensor(INITIAL_GATE))
        nn.init.zeros_(self.scale.weight)
        nn.init.zeros_(self.shift.weight)

    def forward(self, image: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        if image.shape[0] != state.shape[0] or image.shape[2:] != state.shape[2:]:
            raise ValueError(f"image/state grid mismatch: {tuple(image.shape)} vs {tuple(state.shape)}")
        modulation = self.project(state) + self.scale(state) * self.norm(image) + self.shift(state)
        return image + self.gate * modulation


class LocalEncoder(nn.Module):
    """Four levels; image widths 24, 48, 96, 192 (two residual blocks each), state widths 16, 24, 48, 96."""

    def __init__(self, image_widths: Sequence[int] = (24, 48, 96, 192), state_widths: Sequence[int] = (16, 24, 48, 96),
                 blocks_per_level: int = 2, *, image_channels: int = 2, state_channels: int = 12) -> None:
        super().__init__()
        self.image_widths = validate_widths(image_widths, "image widths")
        self.state_widths = validate_widths(state_widths, "state widths")
        if len(self.image_widths) != len(self.state_widths):
            raise ValueError("image and state streams must have the same number of levels")
        if not isinstance(blocks_per_level, int) or blocks_per_level <= 0:
            raise ValueError(f"blocks_per_level must be positive; got {blocks_per_level}")
        self.image_channels, self.state_channels = int(image_channels), int(state_channels)
        self.activation_checkpointing = False
        self.image_stream = nn.ModuleList()
        self.state_stream = nn.ModuleList()
        self.fusions = nn.ModuleList()
        previous_image, previous_state = self.image_channels, self.state_channels
        for level, (image_width, state_width) in enumerate(zip(self.image_widths, self.state_widths)):
            stride = 1 if level == 0 else 2
            self.image_stream.append(EncoderStage(
                nn.Conv3d(previous_image, image_width, 3, stride=stride, padding=1, bias=False),
                [ResidualBlock3d(image_width, image_width) for _ in range(blocks_per_level)]))
            self.state_stream.append(DepthwiseConvBlock3d(previous_state, state_width, stride=stride))
            self.fusions.append(StateFusion3d(image_width, state_width))
            previous_image, previous_state = image_width, state_width

    @property
    def levels(self) -> int:
        return len(self.image_stream)

    def forward_to_level(self, image: torch.Tensor, state: torch.Tensor, max_level: int) -> list[torch.Tensor]:
        """Fused image features of levels 0..``max_level`` (BCZYX each)."""
        if image.ndim != 5 or image.shape[1] != self.image_channels:
            raise ValueError(f"expected image [B,{self.image_channels},Z,Y,X]; got {tuple(image.shape)}")
        if state.ndim != 5 or state.shape[1] != self.state_channels:
            raise ValueError(f"expected state [B,{self.state_channels},Z,Y,X]; got {tuple(state.shape)}")
        if image.shape[0] != state.shape[0] or image.shape[2:] != state.shape[2:]:
            raise ValueError(f"image/state grid mismatch: {tuple(image.shape)} vs {tuple(state.shape)}")
        if not 0 <= max_level < self.levels:
            raise ValueError(f"max_level outside the encoder: {max_level}")
        pyramid: list[torch.Tensor] = []
        for level in range(max_level + 1):
            image = run_checkpointed(self, self.image_stream[level], image)
            state = run_checkpointed(self, self.state_stream[level], state)
            image = run_checkpointed(self, self.fusions[level], image, state)
            pyramid.append(image)
        return pyramid

    def forward(self, image: torch.Tensor, state: torch.Tensor) -> list[torch.Tensor]:
        return self.forward_to_level(image, state, self.levels - 1)

    def feature_affines(self, window_affine_xyz: torch.Tensor) -> list[torch.Tensor]:
        """Affine of each level's grid, from the actual strides of the image-stream convolutions."""
        frames = []
        frame = window_affine_xyz
        for stage in self.image_stream:
            frame = conv_feature_affine(frame, stage.conv)
            frames.append(frame)
        return frames
