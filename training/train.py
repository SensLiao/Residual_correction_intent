"""Train SIRB.

    python training/train.py --config sirb/configs/train/sirb_psma.json [--resume | --resume-from CHECKPOINT] \\
        [--device cuda] [key=value ...]

The merged configuration is written to ``<paths.results>/<run>/config.json``; checkpoints and logs go next to it
(training/README.md). ``--resume`` continues from ``checkpoints/latest.pt`` of the run, ``--resume-from`` from the
given checkpoint; only ``paths`` and ``train.runtime`` may differ from the configuration the checkpoint was trained
with. Overrides come after the options, e.g. ``run=sirb_no_correct_tissue_role model.correct_tissue_role=false
train.runtime.workers=8``. The GPU is chosen with ``CUDA_VISIBLE_DEVICES``; SIGINT or SIGTERM stops after the current
training iteration and a checkpoint.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

import torch  # noqa: E402

from sirb.utils.config import add_config_arguments, load_config  # noqa: E402
from training.trainer import Trainer  # noqa: E402

DEFAULT_CONFIG = "sirb/configs/train/sirb_psma.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    resume = parser.add_mutually_exclusive_group()
    resume.add_argument("--resume", action="store_true", help="continue the run from its checkpoints/latest.pt")
    resume.add_argument("--resume-from", type=Path, default=None, metavar="CHECKPOINT",
                        help="continue the run from this checkpoint")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                        help="torch device (default: cuda when available, else cpu)")
    add_config_arguments(parser, default=DEFAULT_CONFIG)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    path = Path(args.config)
    config = load_config(path if path.is_absolute() or path.exists() else REPOSITORY / path, args.overrides)
    trainer = Trainer(config, device=args.device, resume=args.resume_from or (True if args.resume else None))
    result = trainer.run(handle_signals=True)
    return 0 if result["complete"] else 3


if __name__ == "__main__":
    sys.exit(main())
