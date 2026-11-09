"""Make the patient-level split lists, the drawing styles and the nnU-Net folds of a dataset.

    python tools/make_splits.py [--examinations FILE] [--seed 20260717] [--patients 264 57 57] [--folds 5] \\
        [--output splits] [--overwrite] [key=value ...]

The examinations are the files of the dataset's ``labelsTr/`` (``paths.raw`` and ``data.raw`` of the configuration),
or the ids listed in ``--examinations`` (one per line). An examination id is ``<patient>_<date>``; the patient id is
the id without the date. Three rules of ``sirb.data.psma`` make the files:

* ``split_patients``: the patients are ordered by a fixed SHA-256 ranking with ``--seed``, and the first, next and
  last ``--patients`` patients form the training, validation and test sets, with all their examinations.
* ``assign_drawing_styles``: one scribble drawing style per validation and test examination, dealt in equal thirds.
* ``nnunet_folds``: patient-level ``--folds`` folds over the training and validation examinations, for the
  segmentation model that makes the initial segmentations (the test examinations are in no fold).

Written to ``--output``: ``train.txt``, ``val.txt``, ``test.txt`` (one examination id per line, sorted),
``val_styles.json``, ``test_styles.json`` and ``nnunet_folds.json`` (nnU-Net ``splits_final.json`` format). A file
that exists is not overwritten without ``--overwrite``. With the defaults on the 597 examinations of
PSMA-PET-CT-Lesions this writes the files in ``splits/``; ``timing.txt`` there is a hand-picked list and is not
written.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from sirb.data.psma import (SPLIT_PARTITIONS, RawLayout, assign_drawing_styles, nnunet_folds,  # noqa: E402
                            split_patients)
from sirb.utils.config import add_config_arguments, load_config  # noqa: E402

DEFAULT_CONFIG = REPOSITORY / "sirb" / "configs" / "data" / "psma.json"
DEFAULT_SEED = 20260717
DEFAULT_PATIENTS = (264, 57, 57)
DEFAULT_FOLDS = 5
STYLED = ("val", "test")
FOLDS_FILE = "nnunet_folds.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--examinations", type=Path, default=None,
                        help="file with one examination id per line (default: the files of the dataset's labelsTr/)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="seed of the patient ranking (default: %(default)s)")
    parser.add_argument("--patients", type=int, nargs=3, default=list(DEFAULT_PATIENTS),
                        metavar=("TRAIN", "VAL", "TEST"),
                        help="patients in the training, validation and test sets, adding up to all patients "
                             "(default: %(default)s)")
    parser.add_argument("--folds", type=int, default=DEFAULT_FOLDS,
                        help="nnU-Net folds over the training and validation examinations (default: %(default)s)")
    parser.add_argument("--output", type=Path, default=Path("splits"), help="output folder (default: %(default)s)")
    parser.add_argument("--overwrite", action="store_true", help="replace files that exist")
    add_config_arguments(parser, default=DEFAULT_CONFIG.relative_to(REPOSITORY).as_posix())
    return parser


def dataset_examinations(config) -> list[str]:
    """The examination ids of the dataset's ground-truth folder (``<paths.raw>/<data.raw.labels>``)."""
    layout = RawLayout.from_config(config["data"]["raw"])
    labels = Path(config["paths"]["raw"]) / layout.labels
    if not labels.is_dir():
        raise SystemExit(f"no ground-truth folder {labels}; download the dataset or pass --examinations")
    return sorted(path.name[: -len(layout.extension)] for path in labels.iterdir()
                  if path.is_file() and path.name.endswith(layout.extension))


def listed_examinations(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def split_documents(examinations, *, seed: int, patients, folds: int) -> dict[str, str]:
    """File name -> text of every file the tool writes (module docstring)."""
    splits = split_patients(examinations, seed, patients)
    documents = {f"{name}.txt": "".join(f"{examination}\n" for examination in splits[name])
                 for name in SPLIT_PARTITIONS}
    for name in STYLED:
        documents[f"{name}_styles.json"] = json.dumps(assign_drawing_styles(splits[name]), indent=2,
                                                      sort_keys=True) + "\n"
    pool = splits["train"] + splits["val"]
    documents[FOLDS_FILE] = "[\n" + ",\n".join(json.dumps(fold) for fold in nnunet_folds(pool, folds)) + "\n]\n"
    return documents


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config_path = Path(args.config)
    config = load_config(config_path if config_path.is_absolute() or config_path.exists()
                         else REPOSITORY / config_path, args.overrides)
    examinations = (listed_examinations(args.examinations) if args.examinations is not None
                    else dataset_examinations(config))
    documents = split_documents(examinations, seed=args.seed, patients=args.patients, folds=args.folds)
    existing = sorted(name for name in documents if (args.output / name).exists())
    if existing and not args.overwrite:
        raise SystemExit(f"{args.output} already has {existing}; pass --overwrite to replace them")
    args.output.mkdir(parents=True, exist_ok=True)
    for name, text in documents.items():
        (args.output / name).write_text(text, encoding="utf-8", newline="\n")
    counts = {name: documents[f"{name}.txt"].count("\n") for name in SPLIT_PARTITIONS}
    print(f"{len(examinations)} examinations -> {counts} examinations, {args.folds} folds, in {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
