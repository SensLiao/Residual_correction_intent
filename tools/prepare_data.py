"""Write the examination cache of one split from the dataset and the initial segmentations.

Reads the examinations of ``splits/<split>.txt`` from two folders side by side: the dataset ``paths.raw``
(``SIRB_RAW``: ``imagesTr/``, ``labelsTr/``; names in ``data.raw``) and the initial segmentations
``paths.initial_segmentations`` (``SIRB_INITIAL_SEGMENTATIONS``: ``<examination>.nii.gz``, optionally
``<examination>.npz``). Writes ``SIRB_PREPROCESSED/<data.cache.directory>/`` (``sirb.data.preprocessing``):
normalised PET and CT, the initial segmentation (and its probability when there is one) and the ground truth.
Every written examination adds one row to ``prepare_<split>.jsonl`` there. An examination already in the cache is
refused (``--skip-existing`` skips it), and so is an initial segmentation that equals a non-empty ground truth voxel
for voxel.

The test set is prepared only when asked for with ``--split test``.

Examples::

    python tools/prepare_data.py --split train
    python tools/prepare_data.py --split val --examinations psma_0934d8468a1c7624_2021-01-11
    python tools/prepare_data.py --split test data.cache.directory=examinations_test
    python tools/prepare_data.py --pet-reference        # (mu*, s*) of the cached training set
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from sirb.data.preprocessing import (SPLITS, cache_directory, cached_examinations,  # noqa: E402
                                     pet_reference_from_config, pet_reference_from_examinations,
                                     write_examination_cache)
from sirb.data.psma import RawLayout, patient_id, read_raw_examination, read_split, splits_directory  # noqa: E402
from sirb.data.states import append_rows  # noqa: E402
from sirb.utils.config import add_config_arguments, load_config, save_config  # noqa: E402

DEFAULT_CONFIG = REPOSITORY / "sirb" / "configs" / "data" / "psma.json"
EXIT_OK, EXIT_REFUSED = 0, 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", choices=SPLITS, help="the split list to prepare (test only when given)")
    parser.add_argument("--examinations", nargs="+", default=None, help="only these examinations of the split")
    parser.add_argument("--limit", type=int, default=None, help="only the first N examinations")
    parser.add_argument("--skip-existing", action="store_true", help="skip examinations that are already cached")
    parser.add_argument("--pet-reference", action="store_true",
                        help="print the PET reference (mu*, s*) of the cached training set and compare it with "
                             "data.pet_reference; writes nothing")
    add_config_arguments(parser, default=DEFAULT_CONFIG.relative_to(REPOSITORY).as_posix())
    return parser


def report_pet_reference(config, directory: Path) -> int:
    metas = cached_examinations(directory, "train")
    derived = pet_reference_from_examinations(metas)
    configured = pet_reference_from_config(config)
    print(json.dumps({"training_examinations": len(metas), "derived": {"mu": derived.mu, "s": derived.s},
                      "configured": {"mu": configured.mu, "s": configured.s},
                      "equal": derived == configured}, indent=2))
    return EXIT_OK


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config_path = Path(args.config)
    config = load_config(config_path if config_path.is_absolute() or config_path.exists()
                         else REPOSITORY / config_path, args.overrides)
    data = config["data"]
    directory = cache_directory(config)
    if args.pet_reference:
        return report_pet_reference(config, directory)
    if args.split is None:
        parser.error("--split is required (or --pet-reference)")
    ids = read_split(splits_directory(config, REPOSITORY), args.split)
    if args.examinations is not None:
        unknown = sorted(set(args.examinations) - set(ids))
        if unknown:
            parser.error(f"examinations not in the {args.split} split: {unknown}")
        ids = [examination for examination in ids if examination in set(args.examinations)]
    ids = ids[: args.limit]
    raw_root, layout = Path(config["paths"]["raw"]), RawLayout.from_config(data["raw"])
    initial_root = Path(config["paths"]["initial_segmentations"])
    directory.mkdir(parents=True, exist_ok=True)
    save_config(config, directory / f"prepare_{args.split}_config.json")
    rows_path = directory / f"prepare_{args.split}.jsonl"
    for number, examination in enumerate(ids, 1):
        if args.skip_existing and (directory / "inputs" / examination / "meta.json").is_file():
            print(f"[prepare] {number}/{len(ids)} {examination}: cached already, skipped", flush=True)
            continue
        raw = read_raw_examination(raw_root, initial_root, examination, layout)
        if raw.ground_truth.any() and np.array_equal(raw.initial_segmentation, raw.ground_truth):
            print(f"[prepare] REFUSED: {examination}: the initial segmentation equals the non-empty ground truth "
                  f"voxel for voxel ({int(raw.ground_truth.sum())} voxels)", file=sys.stderr)
            return EXIT_REFUSED
        meta = write_examination_cache(
            directory, examination_id=examination, patient_id=patient_id(examination), split=args.split,
            pet_suv=raw.pet_suv, ct_hu=raw.ct_hu, affine_zyx=raw.affine_zyx, normalization=data["normalization"],
            provenance=raw.provenance, initial_probability=raw.initial_probability,
            initial_segmentation=raw.initial_segmentation, ground_truth=raw.ground_truth,
            segmentation_rule=data["initial_segmentation"]["rule"],
            max_disagreement=data["initial_segmentation"]["max_disagreement"])
        append_rows(rows_path, [{
            "examination_id": examination, "patient_id": meta["patient_id"], "split": meta["split"],
            "shape_zyx": meta["shape_zyx"], "initial_segmentation": meta["initial_segmentation"],
            "pet_normalization": meta["pet_normalization"], "gt_positive": meta["label"]["gt_positive"]}])
        print(f"[prepare] {number}/{len(ids)} {examination}", flush=True)
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
