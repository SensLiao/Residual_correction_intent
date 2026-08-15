#!/usr/bin/env python3
"""Infer structured P2T slots on a tensor manifest with dual provenance.

This is the P2T-facing entrypoint for applying a checkpoint trained on the
controlled matched-state lane to natural OOF tensors.  The implementation is
shared with the evaluator so predictions retain the exact six-class ontology,
joint-defined slot decode, operation metrics, and receipt validation.
"""

from __future__ import annotations

import sys
from pathlib import Path


EVALUATION_ROOT = Path(__file__).resolve().parents[1] / "evaluation"
if str(EVALUATION_ROOT) not in sys.path:
    sys.path.insert(0, str(EVALUATION_ROOT))

from evaluate_petct_p2t import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
