"""Generate the ground-truth-corrected states of one split from the examination cache and its ground truth.

For every cached examination of the split, the simulated user draws on trajectories that start from the initial
segmentation, and the target error is corrected with the ground truth after each scribble
(``sirb.data.states.generate_ground_truth_corrected_states``); every state pair that exists for those states is
stored. Output, under ``SIRB_PREPROCESSED/<data.state_sources.ground_truth_corrected.directory>/``::

    pairs/<examination>/<pair id>.npz, pairs.jsonl, generation_log.jsonl, summary.json

``pairs.jsonl`` is the manifest training reads. Existing output files are never appended to: the tool refuses before
writing anything. ``--shard i/n`` processes the examinations at positions i, i + n, ... of the sorted list and writes
``pairs.shard<i>of<n>.jsonl`` (and the log and summary likewise); ``--merge-shards n`` joins the n shards into the
files above in examination order, the same rows one unsharded run writes. CPU only.

Examples::

    python tools/generate_states.py --split train
    python tools/generate_states.py --split train --shard 0/4      # ... --shard 3/4, then:
    python tools/generate_states.py --split train --merge-shards 4
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from sirb.data.preprocessing import ExaminationCache, cache_directory  # noqa: E402
from sirb.data.states import (append_rows, generate_ground_truth_corrected_states,  # noqa: E402
                              read_pool_rows, write_state_pairs)
from sirb.utils.config import add_config_arguments, load_config, save_config  # noqa: E402

DEFAULT_CONFIG = REPOSITORY / "sirb" / "configs" / "data" / "psma.json"
SPLITS = ("train", "val")
EXIT_OK, EXIT_REFUSED = 0, 2
STEMS = ("pairs", "generation_log")


def parse_shard(text: str) -> tuple[int, int]:
    try:
        index, count = (int(value) for value in str(text).split("/"))
    except ValueError:
        raise argparse.ArgumentTypeError(f"shard actual={text!r}, expected i/n") from None
    if not 0 <= index < count:
        raise argparse.ArgumentTypeError(f"shard actual={text!r}, expected 0 <= i < n")
    return index, count


def shard_suffix(index: int, count: int) -> str:
    return "" if count == 1 else f".shard{index}of{count}"


def output_files(root: Path, suffix: str) -> dict[str, Path]:
    files = {stem: root / f"{stem}{suffix}.jsonl" for stem in STEMS}
    files["summary"] = root / f"summary{suffix}.json"
    return files


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", choices=SPLITS, default="train", help="cached split to generate states for")
    parser.add_argument("--examinations", nargs="+", default=None, help="only these cached examinations")
    parser.add_argument("--limit", type=int, default=None, help="only the first N examinations of the shard")
    parser.add_argument("--shard", type=parse_shard, default=(0, 1), help="i/n: examinations at positions i mod n")
    parser.add_argument("--merge-shards", type=int, default=None, metavar="N", help="join N finished shards")
    add_config_arguments(parser, default=DEFAULT_CONFIG.relative_to(REPOSITORY).as_posix())
    return parser


def merge_shards(root: Path, count: int, order: list[str]) -> int:
    files = output_files(root, "")
    existing = [str(path) for path in files.values() if path.exists()]
    if existing:
        print(f"[states] REFUSED: output files exist already: {existing}", file=sys.stderr)
        return EXIT_REFUSED
    shards = [output_files(root, shard_suffix(index, count)) for index in range(count)]
    missing = [str(path) for shard in shards for path in shard.values() if not path.is_file()]
    if missing:
        print(f"[states] REFUSED: shard files are missing: {missing}", file=sys.stderr)
        return EXIT_REFUSED
    for stem in STEMS:
        groups: dict[str, list[dict]] = collections.defaultdict(list)
        for shard in shards:
            for row in read_pool_rows(shard[stem]):
                groups[row["examination_id"]].append(row)
        append_rows(files[stem], [row for examination in order for row in groups.get(examination, [])])
    counts: collections.Counter = collections.Counter()
    for shard in shards:
        counts.update(json.loads(shard["summary"].read_text(encoding="utf-8"))["counts"])
    summary = {"merged_shards": count, "counts": dict(sorted(counts.items()))}
    files["summary"].write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return EXIT_OK


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config_path = Path(args.config)
    config = load_config(config_path if config_path.is_absolute() or config_path.exists()
                         else REPOSITORY / config_path, args.overrides)
    data = config["data"]
    pair_config, state_config = data["state_pairs"], data["ground_truth_corrected"]
    synthetic = bool(pair_config["synthetic_bridges"])
    if synthetic and args.split != "train":
        parser.error("synthetic bridges are made on the training set only")
    root = Path(config["paths"]["preprocessed"]) / data["state_sources"]["ground_truth_corrected"]["directory"]
    cache = ExaminationCache(cache_directory(config), splits=(args.split,))
    order = cache.examinations(args.split)
    if args.examinations is not None:
        unknown = sorted(set(args.examinations) - set(order))
        if unknown:
            parser.error(f"examinations not cached in the {args.split} split: {unknown}")
        order = [examination for examination in order if examination in set(args.examinations)]
    if args.merge_shards is not None:
        return merge_shards(root, int(args.merge_shards), order)
    index, count = args.shard
    files = output_files(root, shard_suffix(index, count))
    existing = [str(path) for path in files.values() if path.exists()]
    if existing:
        print(f"[states] REFUSED: output files exist already, nothing was written: {existing}", file=sys.stderr)
        return EXIT_REFUSED
    from sirb.interaction.scribble_simulator import ScribbleSimulator  # noqa: PLC0415  (generation only)

    simulator = ScribbleSimulator()
    examinations = [examination for position, examination in enumerate(order) if position % count == index]
    examinations = examinations[: args.limit]
    root.mkdir(parents=True, exist_ok=True)
    save_config(config, root / f"config{shard_suffix(index, count)}.json")
    seed = int(state_config["seed"])
    counters: collections.Counter = collections.Counter()
    for number, examination in enumerate(examinations, 1):
        inputs = cache.inputs(examination)
        truth = cache.ground_truth(examination)
        states, log = generate_ground_truth_corrected_states(
            examination_id=examination, patient_id=inputs.patient_id, split=args.split,
            initial_segmentation=inputs.initial_segmentation, ground_truth=truth, simulator=simulator,
            state_config=state_config, styles=data["drawing_styles"], seed=seed, splits=(args.split,))
        rows, pair_log = write_state_pairs(root, states, truth, seed=seed, pair_config=pair_config,
                                           synthetic_bridges=synthetic, synthetic_splits=("train",),
                                           counters=counters)
        counters["examinations"] += 1
        append_rows(files["pairs"], rows)
        append_rows(files["generation_log"], [dict(row, examination_id=examination) for row in log + pair_log])
        print(f"[states] {number}/{len(examinations)} {examination}: {len(states)} states, {len(rows)} pairs",
              flush=True)
    for stem in STEMS:
        append_rows(files[stem], [])                       # an empty shard still declares its files
    summary = {"split": args.split, "seed": seed, "shard": f"{index}/{count}", "examinations": examinations,
               "counts": dict(sorted(counters.items()))}
    files["summary"].write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: summary[key] for key in ("split", "shard", "counts")}, indent=2, sort_keys=True))
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
