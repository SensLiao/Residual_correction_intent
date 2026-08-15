"""The retired Pilot-3 contract compiler must remain non-operational."""

import sys
from pathlib import Path

import pytest

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "scripts" / "data"))

from build_petct_pilot3_contract import (  # noqa: E402
    Pilot3ContractError,
    compile_pilot3_contract,
)


def test_retired_pilot3_contract_cannot_compile() -> None:
    with pytest.raises(Pilot3ContractError, match="superseded"):
        compile_pilot3_contract({})
