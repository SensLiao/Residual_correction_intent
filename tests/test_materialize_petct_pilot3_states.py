"""The retired filename remains as an explicit fail-closed regression test."""

import sys
from pathlib import Path

import pytest

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "scripts"))

from data.materialize_petct_pilot3_states import (  # noqa: E402
    Pilot3MaterializationError,
    construct_pilot3_states,
)


def test_retired_pilot3_entry_cannot_execute() -> None:
    with pytest.raises(Pilot3MaterializationError, match="legacy provenance"):
        construct_pilot3_states()
