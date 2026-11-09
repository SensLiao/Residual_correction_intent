"""Score the masks written by tools/predict.py with the official metrics and summarise them by patient.

    python tools/evaluate.py --config sirb/configs/eval/five_rounds.json --split val [--run NAME] \\
        [--checkpoint checkpoints/sirb_psma.pt] [--mode standard|fast] [--examination ID ...] [key=value ...]

Reads ``<paths.results>/<run>/predictions/<split>/<examination>.npz`` of every examination of the split; with
``--checkpoint``, missing predictions are made first, exactly as tools/predict.py makes them (in ``--mode``,
``standard`` by default). Each examination's masks
of rounds 0..N are scored against its ground truth (``sirb.evaluation.metrics.score_examination``) and summarised
(``sirb.evaluation.protocol``) with the patient-cluster bootstrap of ``evaluation.bootstrap``. Written under
``<paths.results>/<run>/scores/<split>/``:

    examinations.jsonl   one row per examination: examination, patient, style and the scores of rounds 0..N
    summary.json         Final Dice, nAUC-Dice, nAUC-DMM, final false-positive components and volume, and the Dice
                         and DMM curves over rounds 0..N, averaged by patient and per examination
    config.json          the merged configuration

The test set is read only when ``--split test`` is given.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import numpy as np


def _load_predict_tool():
    """tools/predict.py of this folder (it also puts the repository on sys.path): the tools share its functions."""
    path = Path(__file__).resolve().with_name("predict.py")
    spec = importlib.util.spec_from_file_location("sirb_tools_predict", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


predict = _load_predict_tool()

from sirb.data.geometry import voxel_volume_ml  # noqa: E402
from sirb.evaluation.metrics import DICE, DMM, score_examination  # noqa: E402
from sirb.evaluation.protocol import Bootstrap, round_summaries, summary_table  # noqa: E402
from sirb.interaction.scribble_simulator import ScribbleSimulator  # noqa: E402
from sirb.utils.config import load_config, save_config  # noqa: E402
from sirb.utils.io import atomic_write, json_default, write_json_atomic  # noqa: E402
from sirb.utils.logging import get_logger  # noqa: E402

SCORES = "scores"


def score_rows(examinations: Iterable[str], *, cache: Any, folder: Path,
               log: Callable[[str], None] = print) -> list[dict]:
    """One scored row per examination of the predictions in ``folder`` (module docstring)."""
    rows = []
    for examination_id in examinations:
        arrays, record = predict.load_trajectory(folder, examination_id)
        masks = arrays["masks"]
        truth = np.asarray(cache.ground_truth(examination_id), dtype=bool)
        if masks.shape[1:] != truth.shape:
            raise SystemExit(f"{examination_id}: masks {masks.shape[1:]} and ground truth {truth.shape} differ")
        scores = score_examination(list(masks), truth, voxel_volume_ml=voxel_volume_ml(record["affine_xyz"]))
        rows.append({"examination": examination_id, "patient": record["patient"], "style": record["style"],
                     **scores})
        log(f"{examination_id}: Dice {scores[DICE][0]} -> {scores[DICE][-1]}")
    return rows


def summaries(rows: list[dict], bootstrap: Bootstrap) -> dict:
    """The summary table and the Dice and DMM curves (``sirb.evaluation.protocol``) as JSON values."""
    return {"metrics": {name: summary.to_dict() for name, summary in summary_table(rows, bootstrap=bootstrap).items()},
            "curves": {metric: [summary.to_dict() for summary in round_summaries(rows, metric, bootstrap=bootstrap)]
                       for metric in (DICE, DMM)}}


def write_scores(folder: Path, rows: list[dict], summary: Mapping) -> None:
    """``examinations.jsonl`` (one sorted-key JSON row per line) and ``summary.json``, both published atomically."""
    lines = "".join(json.dumps(row, sort_keys=True, allow_nan=False, default=json_default) + "\n" for row in rows)
    atomic_write(folder / "examinations.jsonl", lambda stream: stream.write(lines.encode("utf-8")))
    write_json_atomic(folder / "summary.json", summary)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    args = predict.add_run_arguments(parser, checkpoint_required=False).parse_args(argv)
    config = load_config(args.config, args.overrides)
    bootstrap = Bootstrap(**config["evaluation"]["bootstrap"])
    examinations = predict.selected_examinations(args, config)
    cache = predict.open_cache(config, args.split)
    run = predict.run_folder(config, args.run)
    predictions = run / predict.PREDICTIONS / args.split
    scores = run / SCORES / args.split
    logger = get_logger(log_file=scores / "evaluate.log")
    missing = [examination for examination in examinations
               if not predict.trajectory_paths(predictions, examination)[1].is_file()]
    if missing:
        if args.checkpoint is None:
            raise SystemExit(f"{len(missing)} examinations have no predictions in {predictions} (first: {missing[0]}); "
                             "run tools/predict.py, or pass --checkpoint to make them here")
        predictor, settings, predictions = predict.prepare(config, args)
        session = settings["session"]
        predict.predict_examinations(missing, cache=cache, predictor=predictor,
                                     simulator=ScribbleSimulator(session["simulator_seed"]),
                                     styles=predict.drawing_styles(args.split, predict.splits_directory(config)),
                                     session=session, folder=predictions,
                                     settings=settings, log=logger.info)
    save_config(config, scores / "config.json")
    rows = score_rows(examinations, cache=cache, folder=predictions, log=logger.info)
    write_scores(scores, rows, summaries(rows, bootstrap))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
