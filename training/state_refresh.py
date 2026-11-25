"""Self-generated states: five-round interactions of a frozen copy of the current model on one half of the training
patients.

At each refresh iteration of ``data.self_generated.refresh`` the trainer saves a checkpoint and runs, for one patient
half (``sirb.data.psma.patient_half``), every training examination of that half through ``data.self_generated.rounds``
correction rounds: the simulated user draws on the current mask in the examination's drawing style
(``sirb.data.psma.assign_drawing_styles`` over the whole training list), a frozen copy of the model predicts with the
options of ``train.refresh.predictor`` (``sirb.sirb_predictor.SIRBPredictor``) and the update rule applies the
correction (``sirb.interaction.session.run_session``). The masks after rounds 1 to 4, each with the scribble the
simulated user drew on it, become self-generated states (``sirb.data.states.states_from_rollout``; the initial
segmentation with the first scribble is left out, the ground-truth-corrected states already hold it), and every state
pair that exists for them is stored (``sirb.data.states.write_state_pairs``). When all examinations of the half are done the pool is published
(``pool.json``) and the trainer's sampler draws from it in the self-generated slots of ``data.mixing``.

A pool lives in ``<paths.preprocessed>/<data.self_generated.directory>/<run>/<pool name>/``::

    examinations/<examination>/attempt-NNN/pairs/...    the pairs of one attempt
    examinations/<examination>/complete.json            rows and round records of the finished attempt (written last)
    pool.json                                           identity, examinations and rows of the complete pool

A refresh that stops part-way resumes: finished examinations are kept and the others start a new attempt.
"""

from __future__ import annotations

import collections
import copy
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch

from sirb.build_sirb import model_settings
from sirb.data.preprocessing import ExaminationCache, cache_directory, pet_reference_from_config
from sirb.data.psma import assign_drawing_styles, patient_half
from sirb.data.states import POOL_MANIFEST, publish_pool, read_pool_rows, states_from_rollout, write_state_pairs
from sirb.interaction.scribble_simulator import ScribbleSimulator
from sirb.interaction.session import Examination, run_session
from sirb.modeling.sirb import SIRB
from sirb.sirb_predictor import PREDICTOR_OPTIONS, SIRBPredictor
from sirb.utils.io import read_json, write_json_atomic

__all__ = ["REFRESH_KEYS", "RefreshPoint", "examination_record", "generate_examination", "generate_pool",
           "publish_refresh", "refresh_examinations", "refresh_points", "refresh_predictor", "refresh_run_tag",
           "self_generated_directory", "self_generated_pool_name"]

#: Keys of ``train.refresh``.
REFRESH_KEYS = {"simulator_seed": int, "draw_on_empty_ground_truth": bool, "predictor": dict}
EXAMINATIONS_FOLDER = "examinations"
COMPLETE_RECORD = "complete.json"
_ATTEMPT = re.compile(r"attempt-(\d+)")


@dataclass(frozen=True)
class RefreshPoint:
    """One refresh: after training ``iteration``, the self-generated states of ``patient_half``."""
    iteration: int
    patient_half: int


def refresh_points(config: Mapping[str, Any]) -> list[RefreshPoint]:
    """The refreshes of ``data.self_generated.refresh``, in order."""
    points = [RefreshPoint(int(point["iteration"]), int(point["patient_half"]))
              for point in config["data"]["self_generated"]["refresh"]]
    for point in points:
        if point.iteration < 1 or point.patient_half not in (0, 1):
            raise ValueError(f"a refresh needs an iteration >= 1 and patient half 0 or 1, not {point}")
    if [point.iteration for point in points] != sorted(point.iteration for point in points):
        raise ValueError("data.self_generated.refresh must be in iteration order")
    return points


def self_generated_pool_name(config: Mapping[str, Any], point: RefreshPoint) -> str:
    """The pool name ``data.self_generated.name`` of a refresh (``sirb.data.pairs.mixing_slots`` uses the same)."""
    return config["data"]["self_generated"]["name"].format(half=point.patient_half, iteration=point.iteration)


def self_generated_directory(config: Mapping[str, Any], run: str, name: str) -> Path:
    """``<paths.preprocessed>/<data.self_generated.directory>/<run>/<name>``."""
    return Path(config["paths"]["preprocessed"]) / config["data"]["self_generated"]["directory"] / run / name


def refresh_run_tag(run: str, iteration: int) -> str:
    """The tag in the ids of the self-generated states of a run's refresh."""
    return f"{run}-self-{iteration}"


def refresh_examinations(roster: Mapping[str, str], half: int) -> list[str]:
    """The training examinations of one patient half, sorted."""
    return sorted(examination for examination, patient in roster.items() if patient_half(patient) == half)


def refresh_predictor(model: SIRB, config: Mapping[str, Any], *, device) -> SIRBPredictor:
    """A predictor on a frozen copy of ``model`` with the options of ``train.refresh.predictor``."""
    settings = model_settings(config)
    options = dict(config["train"]["refresh"]["predictor"])
    unknown = sorted(set(options) - set(PREDICTOR_OPTIONS))
    if unknown:
        raise ValueError(f"train.refresh.predictor: unknown options {unknown}; known: {list(PREDICTOR_OPTIONS)}")
    frozen = copy.deepcopy(model).eval()
    for parameter in frozen.parameters():
        parameter.requires_grad_(False)
    return SIRBPredictor(frozen, grid_spacing_mm=settings["grid_spacing_mm"],
                         whole_volume_shape=settings["whole_volume_shape"], device=device, **options)


def _new_attempt(folder: Path) -> Path:
    numbers = [int(match.group(1)) for match in (_ATTEMPT.fullmatch(item.name) for item in folder.glob("attempt-*"))
               if match]
    number = max(numbers, default=0) + 1
    while True:
        attempt = folder / f"attempt-{number:03d}"
        try:
            attempt.mkdir(parents=True)
            return attempt
        except FileExistsError:
            number += 1


def generate_examination(predictor, cache: ExaminationCache, examination_id: str, *, style: str, simulator,
                         rounds: int, draw_on_empty_ground_truth: bool, run_tag: str, seed: int,
                         pair_config: Mapping[str, Any], root: Path):
    """(manifest rows relative to ``root``, log rows, session result) of one examination's self-generated states;
    its pairs are written under ``root``."""
    inputs = cache.inputs(examination_id)
    truth = np.asarray(cache.ground_truth(examination_id), dtype=bool)
    examination = Examination(inputs.pet, inputs.ct, inputs.affine_xyz, inputs.initial_segmentation,
                              pet_pad_value=inputs.pet_pad_value, examination_id=inputs.examination_id,
                              patient_id=inputs.patient_id)
    session = run_session(predictor, examination, truth, simulator=simulator, style=style, rounds=rounds,
                          draw_on_empty_ground_truth=draw_on_empty_ground_truth)
    states, log = states_from_rollout(examination_id=examination_id, patient_id=inputs.patient_id, split="train",
                                      ground_truth=truth, masks=list(session.masks), scribbles=list(session.scribbles),
                                      correction_types=list(session.correction_types), style=style,
                                      trajectory_index=0, run_tag=run_tag)
    rows, pair_log = write_state_pairs(root, states, truth, seed=seed, pair_config=pair_config,
                                       synthetic_bridges=False, counters=collections.Counter())
    return rows, log + pair_log, session


def examination_record(directory: Path, examination_id: str) -> dict | None:
    """The completion record of an examination of a pool, or None when it is not finished."""
    path = Path(directory) / EXAMINATIONS_FOLDER / examination_id / COMPLETE_RECORD
    return read_json(path) if path.is_file() else None


def generate_pool(model: SIRB, config: Mapping[str, Any], *, run: str, point: RefreshPoint,
                  roster: Mapping[str, str], directory: Path, device, seed: int,
                  log: Callable[[str], None] = print) -> list[dict]:
    """Generate (or finish) the self-generated states of one refresh in ``directory``; returns the pool rows, paths
    relative to ``directory``. Nothing is published."""
    directory = Path(directory)
    data = config["data"]
    refresh = config["train"]["refresh"]
    examinations = refresh_examinations(roster, point.patient_half)
    if not examinations:
        raise ValueError(f"patient half {point.patient_half} has no training examination")
    styles = assign_drawing_styles(sorted(roster))
    cache = ExaminationCache(cache_directory(config), splits=("train",),
                             pet_reference=pet_reference_from_config(config))
    predictor = refresh_predictor(model, config, device=device)
    simulator = ScribbleSimulator(refresh["simulator_seed"])
    run_tag = refresh_run_tag(run, point.iteration)
    rows: list[dict] = []
    for number, examination in enumerate(examinations, 1):
        if cache.meta(examination)["patient_id"] != roster[examination]:
            raise ValueError(f"{examination}: cached patient differs from the training roster")
        record = examination_record(directory, examination)
        if record is None:
            folder = directory / EXAMINATIONS_FOLDER / examination
            attempt = _new_attempt(folder)
            with torch.no_grad():
                found, events, session = generate_examination(
                    predictor, cache, examination, style=styles[examination], simulator=simulator,
                    rounds=int(data["self_generated"]["rounds"]),
                    draw_on_empty_ground_truth=refresh["draw_on_empty_ground_truth"], run_tag=run_tag, seed=seed,
                    pair_config=data["state_pairs"], root=attempt)
            prefix = attempt.relative_to(directory).as_posix()
            found = [dict(row, path=f"{prefix}/{row['path']}") for row in found]
            record = {"examination_id": examination, "patient_id": roster[examination],
                      "style": styles[examination], "attempt": attempt.name, "rows": found, "log": events,
                      "rounds": [round_record.to_dict() for round_record in session.rounds]}
            write_json_atomic(folder / COMPLETE_RECORD, record, sort_keys=False, ensure_ascii=False)
        rows.extend(record["rows"])
        log(f"self-generated states {directory.name}: {number}/{len(examinations)} {examination}, "
            f"{len(record['rows'])} pairs")
    if not rows:
        raise ValueError(f"the refresh of patient half {point.patient_half} gave no state pair")
    return rows


def publish_refresh(directory: Path, rows: Sequence[Mapping[str, Any]], *, run: str, point: RefreshPoint,
                    roster: Mapping[str, str]) -> Path:
    """Publish a complete pool (``pool.json``); a pool already published with the same identity is kept."""
    identity = {"run": run, "iteration": point.iteration, "patient_half": point.patient_half}
    examinations = refresh_examinations(roster, point.patient_half)
    manifest = Path(directory) / POOL_MANIFEST
    if manifest.is_file():
        existing = read_json(manifest)
        if existing.get("identity") != identity or existing.get("examinations") != sorted(examinations):
            raise ValueError(f"{manifest} belongs to another refresh: {existing.get('identity')}")
        return manifest
    return publish_pool(directory, rows, identity=identity, examinations=examinations)


def published_rows(directory: Path) -> list[dict]:
    """The rows of a published pool."""
    return read_pool_rows(Path(directory) / POOL_MANIFEST)
