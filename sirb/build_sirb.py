"""Build SIRB from its configuration and, optionally, load a checkpoint.

The model section of a configuration (``sirb/configs/model/sirb.json``, or one of ``variants/*.json``)::

    window_size              local window in 3 mm cells (112)
    grid_spacing_mm          spacing of the network grid, ZYX (3 mm)          -- used by the input construction
    whole_volume_shape       ZYX cells of the whole-volume grid (96, 64, 64)   -- used by the input construction
    init_seed                seed of the initial weights
    scribble_conditioning    false: the No scribble conditioning variant
    correct_tissue_role      false: the No correct-tissue role variant
    native_refinement        false: the Coarse roles only variant
    merged_non_target_roles  true: the Merged non-target roles variant (training objective only; the trainer reads it)

Runtime settings that change memory and time but not the outputs are keywords of :func:`build_sirb`, not
configuration.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

import torch

from .modeling.sirb import SIRB
from .utils.checkpoint import load_state_dict
from .utils.config import load_config

__all__ = ["MODEL_KEYS", "build_sirb", "model_settings"]

#: Every key of the model section and its type; a missing or unknown key is an error.
MODEL_KEYS = {"window_size": int, "grid_spacing_mm": list, "whole_volume_shape": list, "init_seed": int,
              "scribble_conditioning": bool, "correct_tissue_role": bool, "native_refinement": bool,
              "merged_non_target_roles": bool}


def model_settings(config: Mapping[str, Any]) -> dict[str, Any]:
    """The checked model section of a configuration (a full configuration with a ``model`` key)."""
    if not isinstance(config, Mapping) or not isinstance(config.get("model"), Mapping):
        raise ValueError("the configuration needs a 'model' section (see sirb/configs/model/sirb.json)")
    model = dict(config["model"])
    missing, unknown = sorted(set(MODEL_KEYS) - set(model)), sorted(set(model) - set(MODEL_KEYS))
    if missing or unknown:
        raise ValueError(f"model section: missing keys {missing}, unknown keys {unknown}")
    for key, kind in MODEL_KEYS.items():
        value = model[key]
        if not isinstance(value, kind) or (kind is int and isinstance(value, bool)):
            raise ValueError(f"model.{key} actual={value!r}, expected {kind.__name__}")
    for key in ("grid_spacing_mm", "whole_volume_shape"):
        if len(model[key]) != 3:
            raise ValueError(f"model.{key} needs three ZYX values; got {model[key]!r}")
    return model


def build_sirb(config: Mapping[str, Any] | str | os.PathLike, checkpoint: str | os.PathLike | Mapping | None = None, *,
               device: str | torch.device = "cpu", activation_checkpointing: bool = False, source_chunk: int = 4,
               recompute_role_head: bool = False, strict: bool = True) -> SIRB:
    """SIRB built from ``config`` (a configuration or the path of a configuration file).

    Initial weights depend only on ``model.init_seed`` (the global random state is left unchanged). ``checkpoint``
    is a checkpoint file (``sirb.utils.checkpoint``, read with ``weights_only=True``) or a state dict; its keys
    must match the model exactly unless ``strict`` is false.
    """
    if not isinstance(config, Mapping):
        config = load_config(config)
    settings = model_settings(config)
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(settings["init_seed"])  # the CPU generator only
        model = SIRB(window_size=settings["window_size"], scribble_conditioning=settings["scribble_conditioning"],
                     correct_tissue_role=settings["correct_tissue_role"],
                     native_refinement=settings["native_refinement"],
                     activation_checkpointing=activation_checkpointing, source_chunk=source_chunk,
                     recompute_role_head=recompute_role_head)
    if checkpoint is not None:
        state = checkpoint if isinstance(checkpoint, Mapping) else load_state_dict(checkpoint)
        model.load_state_dict(state, strict=strict)
    return model.to(device)
