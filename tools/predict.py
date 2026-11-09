"""Correct the examinations of one split with SIRB and the simulated user, and write the mask of every round.

    python tools/predict.py --config sirb/configs/eval/five_rounds.json --checkpoint checkpoints/sirb_psma.pt \\
        --split val [--mode standard|fast] [--run NAME] [--examination ID ...] [--device cuda] [key=value ...]

Every examination of the split (or each one given with the repeatable --examination) is corrected for
``session.rounds`` rounds (``sirb.interaction.session.run_session``) in its drawing style
(``<split>_styles.json`` of the split lists, ``data.splits``). The predictor runs in the inference mode ``--mode``
(``standard`` by default, or ``fast``; ``sirb.sirb_predictor.MODES``), and the configuration's ``predictor``
section sets the remaining options or overrides the mode's. Written under
``<paths.results>/<run>/predictions/<split>/``:

    <examination>.npz    masks (rounds 0..N, bool ZYX), scribbles (rounds 1..N, empty when skipped),
                         correction_types (+1 addition, -1 removal, 0 skipped), skipped
    <examination>.json   identifiers, drawing style, affine, round records and the settings of the run
    config.json          the merged configuration

An examination whose two files exist with the same settings is not run again; files with other settings are refused.
The test set is read only when ``--split test`` is given. ``runtime`` sets the CPU threads, TF32 and cuDNN autotuning.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import numpy as np
import torch

REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from sirb.interaction.scribble_simulator import STYLES, ScribbleSimulator  # noqa: E402
from sirb.interaction.session import Examination, SessionResult, run_session  # noqa: E402
from sirb.sirb_predictor import DEFAULT_MODE, MODES, SIRBPredictor  # noqa: E402
from sirb.utils.config import add_config_arguments, load_config, save_config  # noqa: E402
from sirb.utils.io import read_json, read_npz, write_json_atomic, write_npz_atomic  # noqa: E402
from sirb.utils.logging import get_logger  # noqa: E402

SPLITS = ("val", "test")
SPLITS_DIRECTORY = REPOSITORY / "splits"
SESSION_KEYS = {"rounds": int, "simulator_seed": int, "draw_on_empty_ground_truth": bool}
RUNTIME_KEYS = {"tf32": bool, "cudnn_benchmark": bool, "cpu_threads": int}
PREDICTIONS = "predictions"


# ------------------------------------------------------------------------------------------- shared by the tools
def checked_section(config: Mapping[str, Any], name: str, keys: Mapping[str, type]) -> dict:
    """The configuration section ``name`` with exactly ``keys``, each of its type."""
    section = config.get(name)
    if not isinstance(section, Mapping):
        raise ValueError(f"the configuration needs a {name!r} section")
    missing, unknown = sorted(set(keys) - set(section)), sorted(set(section) - set(keys))
    if missing or unknown:
        raise ValueError(f"{name} section: missing keys {missing}, unknown keys {unknown}")
    for key, kind in keys.items():
        value = section[key]
        if not isinstance(value, kind) or (kind is int and isinstance(value, bool)):
            raise ValueError(f"{name}.{key} actual={value!r}, expected {kind.__name__}")
    return dict(section)


def apply_runtime(config: Mapping[str, Any]) -> dict:
    """Set the CPU threads, TF32 and cuDNN autotuning of the ``runtime`` section; returns the section."""
    runtime = checked_section(config, "runtime", RUNTIME_KEYS)
    torch.set_num_threads(runtime["cpu_threads"])
    torch.backends.cuda.matmul.allow_tf32 = runtime["tf32"]
    torch.backends.cudnn.allow_tf32 = runtime["tf32"]
    torch.backends.cudnn.benchmark = runtime["cudnn_benchmark"]
    return runtime


def default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def run_folder(config: Mapping[str, Any], run: str | None) -> Path:
    """``<paths.results>/<run>``; ``run`` defaults to the configuration's ``run``."""
    name = run if run is not None else config.get("run")
    if not isinstance(name, str) or not name or Path(name).name != name:
        raise ValueError(f"the run name must be a plain folder name, got {name!r}")
    return Path(config["paths"]["results"]) / name


def open_cache(config: Mapping[str, Any], split: str):
    """The preprocessed examinations of ``split`` (only that split is readable), PET in the reference coordinates."""
    from sirb.data.preprocessing import ExaminationCache, cache_directory, pet_reference_from_config  # noqa: PLC0415

    return ExaminationCache(cache_directory(config), splits=(split,), pet_reference=pet_reference_from_config(config))


def splits_directory(config: Mapping[str, Any]) -> Path:
    """The split lists: ``data.splits`` of the configuration (``sirb.data.psma.splits_directory``), else
    ``splits/`` of the repository folder."""
    if "splits" not in config.get("data", {}):
        return SPLITS_DIRECTORY
    from sirb.data.psma import splits_directory as configured  # noqa: PLC0415

    return configured(config, REPOSITORY)


def split_examinations(split: str, directory: Path = SPLITS_DIRECTORY) -> list[str]:
    from sirb.data.psma import read_split  # noqa: PLC0415

    return read_split(directory, split)


def drawing_styles(split: str, directory: Path = SPLITS_DIRECTORY) -> dict[str, str]:
    from sirb.data.psma import read_drawing_styles  # noqa: PLC0415

    return read_drawing_styles(directory, split)


def trajectory_paths(folder: Path, examination_id: str) -> tuple[Path, Path]:
    return folder / f"{examination_id}.npz", folder / f"{examination_id}.json"


def save_trajectory(folder: Path, result: SessionResult, examination: Examination, settings: Mapping) -> None:
    """``<examination>.npz`` and ``<examination>.json`` of one session (module docstring); JSON last."""
    archive, record = trajectory_paths(folder, result.examination_id)
    blank = np.zeros_like(result.masks[0])
    write_npz_atomic(archive, {
        "masks": np.stack(result.masks),
        "scribbles": np.stack([blank if scribble is None else scribble for scribble in result.scribbles]),
        "correction_types": np.array([0 if kind is None else kind for kind in result.correction_types],
                                     dtype=np.int8),
        "skipped": np.array(result.skipped, dtype=bool)})
    write_json_atomic(record, {
        "examination": result.examination_id, "patient": result.patient_id, "style": result.style,
        "affine_xyz": np.asarray(examination.affine_xyz, dtype=float).tolist(),
        "rounds": [round_record.to_dict() for round_record in result.rounds], "settings": dict(settings)})


def load_trajectory(folder: Path, examination_id: str) -> tuple[dict, dict]:
    """(arrays, record) of a session written by :func:`save_trajectory`."""
    archive, record = trajectory_paths(folder, examination_id)
    return read_npz(archive), read_json(record)


def finished(folder: Path, examination_id: str, settings: Mapping) -> bool:
    """Whether the examination's session is on disk with ``settings``; other settings are refused."""
    archive, record = trajectory_paths(folder, examination_id)
    if not record.is_file():
        return False
    stored = read_json(record).get("settings")
    if stored != dict(settings):
        raise SystemExit(f"{record} was written with other settings:\n  stored {stored}\n  now    {dict(settings)}\n"
                         "use another run name, or delete that folder first")
    if not archive.is_file():
        raise SystemExit(f"{record} exists without {archive.name}; delete it and run again")
    return True


def session_settings(config: Mapping[str, Any], predictor: SIRBPredictor, checkpoint: str | None,
                     mode: str) -> dict:
    """What decides a session's outputs: checkpoint, model, inference mode, predictor options and session settings."""
    return {"checkpoint": None if checkpoint is None else str(checkpoint), "model": dict(config["model"]),
            "mode": mode, "predictor": predictor.options,
            "session": checked_section(config, "session", SESSION_KEYS)}


def predict_examinations(examinations: Iterable[str], *, cache: Any, predictor: Any, simulator: Any,
                         styles: Mapping[str, str], session: Mapping[str, Any], folder: Path,
                         settings: Mapping[str, Any], log: Callable[[str], None] = print) -> list[str]:
    """Run and save the session of every examination not yet on disk; returns the examinations that ran."""
    folder.mkdir(parents=True, exist_ok=True)
    examinations = list(examinations)
    ran = []
    for number, examination_id in enumerate(examinations, start=1):
        if finished(folder, examination_id, settings):
            continue
        style = styles.get(examination_id)
        if style not in STYLES:
            raise SystemExit(f"{examination_id}: no drawing style in the split's style file")
        started = time.perf_counter()
        examination = Examination.from_inputs(cache.inputs(examination_id))
        result = run_session(predictor, examination, cache.ground_truth(examination_id), simulator=simulator,
                             style=style, rounds=session["rounds"],
                             draw_on_empty_ground_truth=session["draw_on_empty_ground_truth"])
        save_trajectory(folder, result, examination, settings)
        ran.append(examination_id)
        log(f"{number}/{len(examinations)} {examination_id} {style}: {sum(result.skipped)} skipped rounds, "
            f"{time.perf_counter() - started:.1f} s")
    return ran


def prepare(config: Mapping[str, Any], args: argparse.Namespace) -> tuple[SIRBPredictor, dict, Path]:
    """(predictor, settings, predictions folder) of a run; writes the merged configuration."""
    apply_runtime(config)
    predictor = SIRBPredictor.from_config(config, args.checkpoint, mode=args.mode, device=args.device)
    settings = session_settings(config, predictor, args.checkpoint, args.mode)
    folder = run_folder(config, args.run) / PREDICTIONS / args.split
    save_config(config, folder / "config.json")
    return predictor, settings, folder


def add_run_arguments(parser: argparse.ArgumentParser, *, checkpoint_required: bool) -> argparse.ArgumentParser:
    parser.add_argument("--split", required=True, choices=SPLITS,
                        help="the examinations to use; the test set is read only when given explicitly")
    parser.add_argument("--checkpoint", required=checkpoint_required, default=None,
                        help="SIRB checkpoint (training/train.py writes one)")
    parser.add_argument("--mode", choices=list(MODES), default=DEFAULT_MODE,
                        help="inference mode: standard (flip averaging, FP32) or fast (one pass, bfloat16, "
                             "torch.compile); both compute near the scribble (default: %(default)s)")
    parser.add_argument("--run", default=None, help="results folder name under paths.results (default: config 'run')")
    parser.add_argument("--examination", action="append", default=None, metavar="ID", dest="examinations",
                        help="only this examination of the split; repeat for several")
    parser.add_argument("--device", default=default_device(), help="torch device (default: %(default)s)")
    return add_config_arguments(parser)


def selected_examinations(args: argparse.Namespace, config: Mapping[str, Any]) -> list[str]:
    examinations = split_examinations(args.split, splits_directory(config))
    if args.examinations is None:
        return examinations
    unknown = sorted(set(args.examinations) - set(examinations))
    if unknown:
        raise SystemExit(f"not in the {args.split} split: {unknown}")
    return list(args.examinations)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    args = add_run_arguments(parser, checkpoint_required=True).parse_args(argv)
    config = load_config(args.config, args.overrides)
    predictor, settings, folder = prepare(config, args)
    logger = get_logger(log_file=folder / "predict.log")
    session = settings["session"]
    predict_examinations(selected_examinations(args, config), cache=open_cache(config, args.split),
                         predictor=predictor, simulator=ScribbleSimulator(session["simulator_seed"]),
                         styles=drawing_styles(args.split, splits_directory(config)), session=session, folder=folder,
                         settings=settings, log=logger.info)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
