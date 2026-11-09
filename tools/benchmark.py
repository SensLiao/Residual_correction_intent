"""Time SIRB per scribble on the timing examinations (``sirb/configs/eval/speed.json``).

    python tools/benchmark.py --config sirb/configs/eval/speed.json --checkpoint checkpoints/sirb_psma.pt \\
        [--mode standard] [--mode fast] [--run NAME] [--device cuda] [key=value ...]

Times the inference modes of ``sirb.sirb_predictor.MODES`` given with the repeatable ``--mode`` (default: both,
``standard`` then ``fast``); the configuration's ``predictor`` section sets the remaining options of each.
``standard``: local update, left-right flip averaging, FP32; ``fast``: local update, one pass, bfloat16 and
torch.compile. For each mode, in order, the warm-up examination (``speed.warmup_examination``) first runs one whole
session that is not timed (kernel selection, compilation and examination constants); then every examination of
``speed.examinations``, in the listed order, runs a session in its drawing style whose predictor calls are timed: the
wall-clock seconds of each ``predict_state`` call (one per valid scribble), with the device synchronised before and
after the call.

Readings over the calls: all calls; warm calls, every call except each examination's first; warm addition and warm
removal calls; first calls. Each gives the number of calls, median, 90th percentile, mean and maximum. Written under
``<paths.results>/<run>/speed/<mode>/``: ``timing.json`` (readings, every call, settings and environment), the timed
sessions in ``predictions/`` and the warm-up session in ``warmup/`` (the files of tools/predict.py), so every mask
can be compared with another mode or an earlier run.
"""

from __future__ import annotations

import argparse
import importlib.util
import platform
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch


def _load_predict_tool():
    """tools/predict.py of this folder (it also puts the repository on sys.path): the tools share its functions."""
    path = Path(__file__).resolve().with_name("predict.py")
    spec = importlib.util.spec_from_file_location("sirb_tools_predict", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


predict = _load_predict_tool()

from sirb.interaction.scribble_simulator import ScribbleSimulator  # noqa: E402
from sirb.interaction.session import Examination, run_session  # noqa: E402
from sirb.sirb_predictor import MODES, SIRBPredictor  # noqa: E402
from sirb.utils.config import add_config_arguments, load_config, save_config  # noqa: E402
from sirb.utils.io import write_json_atomic  # noqa: E402
from sirb.utils.logging import get_logger  # noqa: E402

SPEED = "speed"
WARMUP = "warmup"
READINGS = ("all", "warm", "warm_addition", "warm_removal", "first")


def statistics(values: Iterable[float]) -> dict | None:
    """Number, median, 90th percentile, mean and maximum of seconds; None without values."""
    seconds = np.asarray(list(values), dtype=np.float64)
    if not seconds.size:
        return None
    return {"n": int(seconds.size), "median": float(np.median(seconds)), "p90": float(np.percentile(seconds, 90)),
            "mean": float(seconds.mean()), "max": float(seconds.max())}


def latency(calls: list[Mapping]) -> dict:
    """The readings of the module docstring; a call is warm unless it is the first call of its examination."""
    seen, warm, first = set(), [], []
    for call in calls:
        (warm if call["examination"] in seen else first).append(call)
        seen.add(call["examination"])
    return {"all": statistics(call["seconds"] for call in calls),
            "warm": statistics(call["seconds"] for call in warm),
            "warm_addition": statistics(call["seconds"] for call in warm if call["correction_type"] == 1),
            "warm_removal": statistics(call["seconds"] for call in warm if call["correction_type"] == -1),
            "first": statistics(call["seconds"] for call in first)}


class TimedPredictor:
    """Forwards ``predict_state`` and records the synchronised wall-clock seconds of every call."""

    def __init__(self, predictor: Any, device: torch.device, calls: list[dict]) -> None:
        self.predictor, self.device, self.calls = predictor, torch.device(device), calls

    def _synchronize(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def predict_state(self, state):
        self._synchronize()
        started = time.perf_counter()
        prediction = self.predictor.predict_state(state)
        self._synchronize()
        seconds = time.perf_counter() - started
        self.calls.append({"examination": state.examination_id, "correction_type": int(state.correction_type),
                           "scribble_voxels": int(state.scribble_zyx.sum()), "seconds": seconds,
                           "applied_voxels": int(prediction.applied.sum())})
        return prediction


def speed_section(config: Mapping[str, Any]) -> dict:
    speed = config.get(SPEED)
    if not isinstance(speed, Mapping) or set(speed) != {"split", "warmup_examination", "examinations"}:
        raise ValueError("the configuration needs a 'speed' section with split, warmup_examination, examinations")
    return dict(speed)


def run_mode(mode: str, *, config: Mapping[str, Any], checkpoint: str, device: str, cache: Any,
             styles: Mapping[str, str], folder: Path, log=print) -> dict:
    """Warm-up session, then the timed sessions of one mode; returns the timing document (also written)."""
    speed = speed_section(config)
    predictor = SIRBPredictor.from_config(config, checkpoint, mode=mode, device=device)
    session = predict.checked_section(config, "session", predict.SESSION_KEYS)
    simulator = ScribbleSimulator(session["simulator_seed"])
    settings = predict.session_settings(config, predictor, checkpoint, mode)

    def correct(examination_id: str, timed: Any, target: Path):
        examination = Examination.from_inputs(cache.inputs(examination_id))
        result = run_session(timed, examination, cache.ground_truth(examination_id), simulator=simulator,
                             style=styles[examination_id], rounds=session["rounds"],
                             draw_on_empty_ground_truth=session["draw_on_empty_ground_truth"])
        target.mkdir(parents=True, exist_ok=True)
        predict.save_trajectory(target, result, examination, settings)
        return result
    with torch.no_grad():
        correct(speed["warmup_examination"], predictor, folder / WARMUP)
        calls: list[dict] = []
        timed = TimedPredictor(predictor, predictor.device, calls)
        started = time.perf_counter()
        for examination_id in speed["examinations"]:
            correct(examination_id, timed, folder / predict.PREDICTIONS)
            log(f"{mode} {examination_id}: {sum(1 for call in calls if call['examination'] == examination_id)} calls")
        wall = time.perf_counter() - started
    document = {"mode": mode, "options": predictor.options, "settings": settings, "latency": latency(calls),
                "calls": calls, "wall_seconds": wall, "environment": environment(predictor.device)}
    write_json_atomic(folder / "timing.json", document)
    return document


def environment(device: torch.device) -> dict:
    found = {"torch": torch.__version__, "cuda": torch.version.cuda, "python": platform.python_version(),
             "cpu_threads": torch.get_num_threads(), "device": str(device)}
    if device.type == "cuda":
        found.update(gpu=torch.cuda.get_device_name(device), cudnn=torch.backends.cudnn.version())
    return found


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, help="SIRB checkpoint")
    parser.add_argument("--mode", action="append", choices=list(MODES), default=None,
                        help="an inference mode to time; repeat for several (default: standard, then fast)")
    parser.add_argument("--run", default=None, help="results folder name under paths.results (default: config 'run')")
    parser.add_argument("--device", default=predict.default_device(), help="torch device (default: %(default)s)")
    args = add_config_arguments(parser).parse_args(argv)
    config = load_config(args.config, args.overrides)
    speed = speed_section(config)
    modes = list(MODES) if args.mode is None else list(dict.fromkeys(args.mode))
    predict.apply_runtime(config)
    cache = predict.open_cache(config, speed["split"])
    styles = predict.drawing_styles(speed["split"], predict.splits_directory(config))
    root = predict.run_folder(config, args.run) / SPEED
    save_config(config, root / "config.json")
    logger = get_logger(log_file=root / "benchmark.log")
    for mode in modes:
        document = run_mode(mode, config=config, checkpoint=args.checkpoint, device=args.device, cache=cache,
                            styles=styles, folder=root / mode, log=logger.info)
        warm = document["latency"]["warm"]
        logger.info(f"{mode}: warm median {None if warm is None else round(warm['median'], 3)} s per scribble "
                    f"over {0 if warm is None else warm['n']} calls")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
