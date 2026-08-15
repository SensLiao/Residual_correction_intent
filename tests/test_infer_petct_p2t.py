"""Current inference entry exposes only the simple-first six-class model."""

import importlib
import sys
from pathlib import Path

import pytest

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "scripts"))

from common.petct_models import (  # noqa: E402
    P2T_DEFERRED_CROSS_ATTENTION_ARCHITECTURE_ID,
    P2T_PRIMARY_ARCHITECTURE_ID,
    build_p2t_model,
)


def test_inference_model_factory_is_six_class_simple_first_only() -> None:
    model = build_p2t_model()
    assert model.architecture_id == P2T_PRIMARY_ARCHITECTURE_ID
    assert model.joint_head.out_features == 6
    assert model.operation_head.out_features == 2
    with pytest.raises(ValueError):
        build_p2t_model(P2T_DEFERRED_CROSS_ATTENTION_ARCHITECTURE_ID)


def test_inference_entry_imports_without_deferred_model_aliases() -> None:
    module = importlib.import_module("p2t.infer_petct_p2t")
    assert callable(module.main)
