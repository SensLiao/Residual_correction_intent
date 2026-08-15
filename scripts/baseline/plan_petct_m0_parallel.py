#!/usr/bin/env python3
"""Print the safe two-GPU five-fold schedule; never launches a job."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from common.petct_route_a_core import ContractError, plan_gpu_queues  # noqa: E402


def _folds(value: str):
    if not value:
        return []
    try:
        return [int(item) for item in value.split(",") if item != ""]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("fold list must contain comma-separated integers") from exc


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--running", type=_folds, default=[])
    parser.add_argument("--completed", type=_folds, default=[])
    parser.add_argument(
        "--running-gpu", action="append", default=[], metavar="FOLD:GPU",
        help="assign each running fold to GPU 0 or 1",
    )
    args = parser.parse_args()
    mapping = {}
    for raw in args.running_gpu:
        try:
            fold, gpu = (int(item) for item in raw.split(":", 1))
        except (TypeError, ValueError):
            parser.error("--running-gpu must be FOLD:GPU")
        mapping[fold] = gpu
    try:
        plan = plan_gpu_queues(
            running_folds=args.running,
            completed_folds=args.completed,
            gpu_for_running=mapping,
        )
    except ContractError as exc:
        parser.error(str(exc))
    print(json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
