"""Deferred architecture arms are not part of the current v2 execution set."""

import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "scripts"))

from common.petct_models import (  # noqa: E402
    P2T_ARCHITECTURE_IDS,
    P2T_DEFERRED_CROSS_ATTENTION_ARCHITECTURE_ID,
    P2T_PRIMARY_ARCHITECTURE_ID,
)


def test_deferred_architecture_has_no_current_checkpoint_validation_arm() -> None:
    assert P2T_ARCHITECTURE_IDS == (P2T_PRIMARY_ARCHITECTURE_ID,)
    assert P2T_DEFERRED_CROSS_ATTENTION_ARCHITECTURE_ID not in P2T_ARCHITECTURE_IDS
