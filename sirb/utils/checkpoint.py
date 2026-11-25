"""Checkpoints of SIRB.

A checkpoint file holds ``{"model": state_dict, "config": dict, "iteration": int}``, where ``config`` is the merged
configuration of the run (JSON values only) and ``iteration`` the number of completed training iterations, plus any
extra entries a trainer adds (for example the optimizer state). A file may also hold a plain state dict; loading it
gives ``config`` and ``iteration`` ``None``.

Files are read with ``torch.load(weights_only=True)`` by default, which refuses arbitrary pickled objects; pass
``weights_only=False`` only for a file you trust.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

import torch

from .io import atomic_write

__all__ = ["CHECKPOINT_KEYS", "load_checkpoint", "load_state_dict", "save_checkpoint", "save_state_dict"]

#: The entries every checkpoint holds.
CHECKPOINT_KEYS = ("model", "config", "iteration")


def _state_dict_of(model: torch.nn.Module | Mapping[str, torch.Tensor]) -> Mapping[str, torch.Tensor]:
    return model.state_dict() if isinstance(model, torch.nn.Module) else model


def _is_state_dict(value: Any) -> bool:
    return (isinstance(value, Mapping) and len(value) > 0
            and all(isinstance(key, str) and isinstance(tensor, torch.Tensor) for key, tensor in value.items()))


def save_checkpoint(path: str | os.PathLike, model: torch.nn.Module | Mapping[str, torch.Tensor], *,
                    config: Mapping[str, Any], iteration: int, **extra: Any) -> Path:
    """Write ``{"model", "config", "iteration", **extra}`` atomically; ``model`` is a module or its state dict."""
    try:
        json.dumps(config, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError(f"the checkpoint config must hold JSON values only: {error}") from error
    if not isinstance(iteration, int) or isinstance(iteration, bool) or iteration < 0:
        raise ValueError(f"iteration must be a non-negative int, not {iteration!r}")
    payload = {"model": _state_dict_of(model), "config": config, "iteration": iteration, **extra}
    return atomic_write(path, lambda stream: torch.save(payload, stream))


def save_state_dict(path: str | os.PathLike, model: torch.nn.Module | Mapping[str, torch.Tensor]) -> Path:
    """Write the plain state dict of ``model`` atomically."""
    state = _state_dict_of(model)
    return atomic_write(path, lambda stream: torch.save(state, stream))


def load_checkpoint(path: str | os.PathLike, *, map_location: Any = "cpu", weights_only: bool = True) -> dict:
    """``{"model", "config", "iteration", ...}`` of a checkpoint file, or of a plain state-dict file with ``config``
    and ``iteration`` ``None`` (module doc)."""
    payload = torch.load(os.fspath(path), map_location=map_location, weights_only=weights_only)
    if _is_state_dict(payload):
        return {"model": payload, "config": None, "iteration": None}
    if isinstance(payload, Mapping) and all(key in payload for key in CHECKPOINT_KEYS) \
            and _is_state_dict(payload["model"]):
        return dict(payload)
    found = sorted(payload) if isinstance(payload, Mapping) else type(payload).__name__
    raise ValueError(f"{path} is not a SIRB checkpoint {CHECKPOINT_KEYS} nor a plain state dict: found {found}")


def load_state_dict(path: str | os.PathLike, *, map_location: Any = "cpu", weights_only: bool = True) -> dict:
    """The model state dict of a checkpoint or plain state-dict file."""
    return load_checkpoint(path, map_location=map_location, weights_only=weights_only)["model"]
