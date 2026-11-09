"""The optimizer and the learning-rate schedule of SIRB training.

AdamW (``train.optimizer``: learning rate, betas, eps, weight decay) on every parameter. The learning rate of
training iteration ``i`` (1-based, ``train.learning_rate``) rises linearly from 0 to the peak over the warm-up
iterations, then follows cosine segments, each from the value its predecessor ended at to its own end value::

    lr(i) = peak * i / warmup                                       for i <= warmup
    lr(i) = b + (a - b) * (1 + cos(pi * (i - s) / (e - s))) / 2     for s < i <= e, one segment [s, e] from a to b

The paper's recipe: warm-up to 2e-4 over 1,000 iterations, cosine to 5e-5 at 40,000 and to 2e-5 at 50,000.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

import torch

__all__ = ["OPTIMIZER_KEYS", "SCHEDULE_KEYS", "check_schedule", "learning_rate", "make_optimizer", "set_learning_rate"]

#: Keys of ``train.optimizer``.
OPTIMIZER_KEYS = ("lr", "betas", "eps", "weight_decay")
#: Keys of ``train.learning_rate``.
SCHEDULE_KEYS = ("warmup_iterations", "cosine_segments")


def make_optimizer(parameters, settings: Mapping[str, Any]) -> torch.optim.AdamW:
    """AdamW with the ``train.optimizer`` settings; ``lr`` is the peak of the schedule."""
    missing, unknown = sorted(set(OPTIMIZER_KEYS) - set(settings)), sorted(set(settings) - set(OPTIMIZER_KEYS))
    if missing or unknown:
        raise ValueError(f"train.optimizer: missing keys {missing}, unknown keys {unknown}")
    return torch.optim.AdamW(parameters, lr=settings["lr"], betas=tuple(settings["betas"]), eps=settings["eps"],
                             weight_decay=settings["weight_decay"])


def check_schedule(schedule: Mapping[str, Any], *, peak: float) -> tuple[int, list[tuple[int, float]]]:
    """(warm-up iterations, cosine segments as (end iteration, end value)) of ``train.learning_rate``, checked."""
    missing, unknown = sorted(set(SCHEDULE_KEYS) - set(schedule)), sorted(set(schedule) - set(SCHEDULE_KEYS))
    if missing or unknown:
        raise ValueError(f"train.learning_rate: missing keys {missing}, unknown keys {unknown}")
    warmup = schedule["warmup_iterations"]
    if isinstance(warmup, bool) or not isinstance(warmup, int) or warmup < 1:
        raise ValueError(f"train.learning_rate.warmup_iterations actual={warmup!r}, expected an integer >= 1")
    if not math.isfinite(float(peak)) or float(peak) <= 0:
        raise ValueError(f"the peak learning rate actual={peak!r}, expected a positive number")
    segments, start = [], warmup
    for entry in schedule["cosine_segments"]:
        if not isinstance(entry, (list, tuple)) or len(entry) != 2:
            raise ValueError(f"a cosine segment is [end iteration, end value], not {entry!r}")
        end, value = entry
        if isinstance(end, bool) or not isinstance(end, int) or end <= start:
            raise ValueError(f"cosine segment ends must increase after the warm-up: {end!r} after {start}")
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"a cosine segment end value must be a finite number >= 0, not {value!r}")
        segments.append((end, value))
        start = end
    if not segments:
        raise ValueError("train.learning_rate needs at least one cosine segment")
    return warmup, segments


def _cosine(start_value: float, end_value: float, fraction: float) -> float:
    return end_value + (start_value - end_value) * (1 + math.cos(math.pi * fraction)) / 2


def learning_rate(schedule: Mapping[str, Any], peak: float, iteration: int) -> float:
    """The learning rate of training ``iteration`` (1 to the last segment end; module docstring)."""
    warmup, segments = check_schedule(schedule, peak=peak)
    last = segments[-1][0]
    if isinstance(iteration, bool) or not isinstance(iteration, int) or not 1 <= iteration <= last:
        raise ValueError(f"training iteration actual={iteration!r}, expected 1..{last}")
    if iteration <= warmup:
        return peak * iteration / warmup
    start, value = warmup, peak
    for end, target in segments:
        if iteration <= end:
            return _cosine(value, target, (iteration - start) / (end - start))
        start, value = end, target
    raise AssertionError("unreachable: the iteration lies inside the checked schedule")


def set_learning_rate(optimizer: torch.optim.Optimizer, value: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = value
