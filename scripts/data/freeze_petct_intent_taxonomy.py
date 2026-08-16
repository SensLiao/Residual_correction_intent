#!/usr/bin/env python3
"""Validate and freeze the six-class intent taxonomy before cue materialization."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any


SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from common.petct_route_a_core import LEGAL_JOINT_GOALS  # noqa: E402


SCHEMA_VERSION = "PETCT-INTENT-TAXONOMY-v2.0"


def canonical_taxonomy() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "operations": ["ADD", "REMOVE"],
        "targets": ["SAME", "NEW"],
        "scopes": ["LOCAL", "COMPLETE"],
        "legal_joint_goals": list(LEGAL_JOINT_GOALS),
        "forbidden_joint_goals": [
            "ADD_NEW_LOCAL",
            "REMOVE_NEW_LOCAL",
        ],
        "cue_channels": ["FG_POSITIVE", "BG_NEGATIVE"],
        "illegal_policy": (
            "hard-fail; never silently coerce, train, score or use as an intervention target"
        ),
        "scope_observation_plane": "prompted axial slice",
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze_taxonomy(config_path: Path) -> dict[str, Any]:
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("experiment config must be valid UTF-8 JSON") from exc
    expected = canonical_taxonomy()
    if config.get("intent_ontology") != expected:
        raise RuntimeError(
            "config.intent_ontology must exactly equal the six-class canonical taxonomy"
        )
    encoded = json.dumps(
        expected, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return {
        "schema_version": "PETCT-INTENT-TAXONOMY-FREEZE-v2.0",
        "status": "PASS",
        "taxonomy": expected,
        "taxonomy_sha256": hashlib.sha256(encoded).hexdigest(),
        "experiment_config": str(config_path.resolve()),
        "experiment_config_sha256": _sha256(config_path),
        "legacy_contract_policy": (
            "PETCT-INTENT-v1.x SAME/NEW ADD-only artifacts are provenance-only "
            "and require explicit offline migration"
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("output already exists")
    try:
        receipt = freeze_taxonomy(args.experiment_config)
    except RuntimeError as exc:
        parser.error(str(exc))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    with args.output.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
