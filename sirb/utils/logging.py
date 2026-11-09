"""Logging: the standard ``logging`` logger of SIRB and a JSON-lines metrics writer."""

from __future__ import annotations

import json
import logging
import math
import os
from pathlib import Path
from typing import Any, Mapping

from .io import json_default

__all__ = ["LOG_FORMAT", "LOGGER_NAME", "MetricsWriter", "get_logger", "read_metrics"]

LOGGER_NAME = "sirb"
LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def get_logger(name: str = LOGGER_NAME, *, level: int = logging.INFO,
               log_file: str | os.PathLike | None = None) -> logging.Logger:
    """The named logger, writing to stderr and, when ``log_file`` is given, appending to that UTF-8 file.

    Calling it again for the same name and file adds no second handler. The logger does not propagate to the root
    logger, so a line is printed once.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False
    formatter = logging.Formatter(LOG_FORMAT)
    if not any(type(handler) is logging.StreamHandler for handler in logger.handlers):
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
    if log_file is not None:
        path = Path(log_file).resolve()
        open_files = {Path(handler.baseFilename) for handler in logger.handlers
                      if isinstance(handler, logging.FileHandler)}
        if path not in open_files:
            path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(path, mode="a", encoding="utf-8")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
    return logger


class MetricsWriter:
    """Append one JSON object per line to ``path`` (UTF-8, LF, sorted keys); each row is on disk when ``write``
    returns. NumPy scalars are written as numbers. NaN and infinity are refused: write ``None`` instead, or stop the
    run that produced them."""

    def __init__(self, path: str | os.PathLike) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, row: Mapping[str, Any] | None = None, **fields: Any) -> None:
        record = {**(row or {}), **fields}
        try:
            line = json.dumps(record, sort_keys=True, ensure_ascii=False, allow_nan=False, default=json_default)
        except ValueError as error:
            raise ValueError(f"metrics row has non-finite values for {_non_finite_keys(record)}: {error}") from error
        with open(self.path, "a", encoding="utf-8", newline="\n") as stream:
            stream.write(line + "\n")


def _non_finite_keys(record: Mapping[str, Any]) -> list[str]:
    keys = []
    for key, value in record.items():
        if isinstance(value, (str, bytes)):
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(number):
            keys.append(key)
    return sorted(keys)


def read_metrics(path: str | os.PathLike) -> list[dict]:
    """Every row of a JSON-lines metrics file."""
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
