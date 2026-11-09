"""The SIRB training loop.

One training iteration (:func:`train_iteration`) trains on two units, a state-pair unit and a scribble-pair unit of
four views each (``sirb.data.pairs``): every view's forward and backward pass in turn, its loss divided by the number
of views (``training.model.sirb_train``; the two windows of a state share its context), then gradient clipping to
``train.gradient_clip_norm`` and one AdamW step with the learning rate of the iteration (``training.optimizer``).

:class:`Trainer` runs ``train.iterations`` iterations:

* units are reserved in iteration order by ``sirb.data.pairs.TrainingPairSampler`` (named random streams and the
  training course) and prepared ``train.runtime.prefetch_depth`` iterations ahead in ``train.runtime.workers`` spawned
  processes (0: in this process). Reservation is the only consumer of the streams and the course, so the units of
  an iteration do not depend on the workers or the depth;
* at each iteration of ``data.self_generated.refresh`` the trainer saves a checkpoint and generates the
  self-generated states of one patient half with a frozen copy of the model (``training.state_refresh``); later
  iterations draw from them;
* a checkpoint is saved every ``train.checkpoint_every`` iterations, at each refresh, at the end and when the process
  is asked to stop (SIGINT, SIGTERM). It holds the model, the optimizer, the random streams and the course as they
  were right after the iteration was reserved, the published self-generated pools and the timings, so a resumed run
  continues bit for bit as if it had not stopped (the same configuration apart from ``train.runtime`` and ``paths``;
  on a GPU, up to its nondeterminism);
* every ``train.log_every`` iterations one row goes to ``logs/train.jsonl``.

The experiment folder is ``<paths.results>/<run>``: ``config.json``, ``checkpoints/latest.pt``,
``checkpoints/iteration_NNNNNN.pt`` (at each refresh and the end), ``logs/train.log``, ``logs/train.jsonl``,
``complete.json``.
"""

from __future__ import annotations

import copy
import multiprocessing
import signal
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from sirb.build_sirb import build_sirb, model_settings
from sirb.data.pairs import (STATE_PAIR, SCRIBBLE_PAIR, RandomStreams, TrainingPairSampler, UnitContext, prepare_unit,
                             read_training_pools)
from sirb.data.preprocessing import cache_directory, pet_reference_from_config
from sirb.data.psma import patient_id, read_split, splits_directory
from sirb.data.states import POOL_MANIFEST
from sirb.utils.checkpoint import load_checkpoint, save_checkpoint
from sirb.utils.config import save_config
from sirb.utils.io import write_json_atomic
from sirb.utils.logging import MetricsWriter, get_logger
from training.loss_fns import LOSS_WEIGHT_KEYS, source_targets
from training.model.sirb_train import SIRBTrain, plain_readings
from training.optimizer import check_schedule, learning_rate, make_optimizer, set_learning_rate
from training.state_refresh import (REFRESH_KEYS, RefreshPoint, generate_pool, publish_refresh, refresh_points,
                                    self_generated_directory, self_generated_pool_name)

__all__ = ["CHECKPOINTS", "LATEST", "RUNTIME_KEYS", "TRAIN_KEYS", "IterationPrefetch", "Trainer", "UnitPreparer",
           "experiment_folder", "pools_in_use", "recipe", "train_iteration", "train_settings", "training_roster"]

REPOSITORY = Path(__file__).resolve().parents[1]
CHECKPOINTS, LOGS = "checkpoints", "logs"
LATEST = "latest.pt"
#: Order of the units of an iteration (the state-pair unit's four views, then the scribble-pair unit's).
UNIT_ORDER = (STATE_PAIR, SCRIBBLE_PAIR)
#: Every key of ``train`` and its type.
TRAIN_KEYS = {"seed": int, "iterations": int, "views_per_iteration": int, "tf32": bool, "optimizer": dict,
              "learning_rate": dict, "gradient_clip_norm": (int, float), "loss": dict, "checkpoint_every": int,
              "log_every": int, "refresh": dict, "runtime": dict}
#: Every key of ``train.runtime`` and its type: machine settings that change memory and time, not the recipe.
RUNTIME_KEYS = {"workers": int, "prefetch_depth": int, "cpu_threads": int, "cudnn_benchmark": bool,
                "voxel_chunk": int, "recompute_convolutions": bool, "recompute_native_voxels": bool,
                "source_chunk": int, "recompute_role_head": bool}
#: Keys of ``train.loss``.
LOSS_KEYS = LOSS_WEIGHT_KEYS + ("dice_epsilon",)
NONFINITE_VIEW_LOSS = "non-finite or non-scalar view loss"


# ------------------------------------------------------------------------------------------- configuration
def _check_section(section: Any, keys: Mapping[str, Any], name: str) -> dict:
    if not isinstance(section, Mapping):
        raise ValueError(f"the configuration needs a {name!r} section")
    missing, unknown = sorted(set(keys) - set(section)), sorted(set(section) - set(keys))
    if missing or unknown:
        raise ValueError(f"{name}: missing keys {missing}, unknown keys {unknown}")
    for key, kind in keys.items():
        value = section[key]
        if isinstance(value, bool) and kind is not bool or not isinstance(value, kind):
            raise ValueError(f"{name}.{key} actual={value!r}, expected {kind}")
    return dict(section)


def train_settings(config: Mapping[str, Any]) -> dict:
    """The checked ``train`` section of a merged configuration."""
    settings = _check_section(config.get("train"), TRAIN_KEYS, "train")
    settings["runtime"] = _check_section(settings["runtime"], RUNTIME_KEYS, "train.runtime")
    settings["refresh"] = _check_section(settings["refresh"], REFRESH_KEYS, "train.refresh")
    loss = _check_section(settings["loss"], {key: (int, float) for key in LOSS_KEYS}, "train.loss")
    if not loss["native_ce"] > 0 or min(loss.values()) < 0:
        raise ValueError(f"train.loss actual={loss}: weights must be >= 0, the native CE weight positive")
    for key in ("iterations", "views_per_iteration", "checkpoint_every", "log_every"):
        if settings[key] < 1:
            raise ValueError(f"train.{key} must be at least 1, got {settings[key]}")
    if settings["views_per_iteration"] != 8:
        raise ValueError("an iteration trains on two units of four views: train.views_per_iteration must be 8")
    runtime = settings["runtime"]
    for key in ("prefetch_depth", "cpu_threads", "voxel_chunk", "source_chunk"):
        if runtime[key] < 1:
            raise ValueError(f"train.runtime.{key} must be at least 1, got {runtime[key]}")
    if runtime["workers"] < 0:
        raise ValueError(f"train.runtime.workers must be >= 0, got {runtime['workers']}")
    _, segments = check_schedule(settings["learning_rate"], peak=settings["optimizer"]["lr"])
    if segments[-1][0] < settings["iterations"]:
        raise ValueError(f"the learning-rate schedule ends at {segments[-1][0]}, before train.iterations "
                         f"{settings['iterations']}")
    if float(settings["gradient_clip_norm"]) <= 0:
        raise ValueError("train.gradient_clip_norm must be positive")
    return settings


def recipe(config: Mapping[str, Any]) -> dict:
    """The configuration without what may change on resuming: ``paths`` and ``train.runtime``."""
    kept = copy.deepcopy(dict(config))
    kept.pop("paths", None)
    if isinstance(kept.get("train"), dict):
        kept["train"].pop("runtime", None)
    return kept


def experiment_folder(config: Mapping[str, Any]) -> Path:
    """``<paths.results>/<run>``."""
    name = config.get("run")
    if not isinstance(name, str) or not name or Path(name).name != name or name in (".", ".."):
        raise ValueError(f"the configuration's run must be a plain folder name, got {name!r}")
    return Path(config["paths"]["results"]) / name


def training_roster(config: Mapping[str, Any]) -> dict[str, str]:
    """{examination: patient} of the training split list (``data.splits``)."""
    examinations = read_split(splits_directory(config, REPOSITORY), "train")
    return {examination: patient_id(examination) for examination in examinations}


def pools_in_use(config: Mapping[str, Any]) -> dict:
    """The configuration whose ``data.state_sources`` keeps only the pools ``data.mixing`` draws from: the
    ground-truth-corrected pool and the earlier-version pools (without these, their slots go to the
    ground-truth-corrected pool and their states need not exist)."""
    mixing = config["data"]["mixing"]
    used = [mixing["ground_truth_corrected_pool"], *mixing["earlier_version_pools"]]
    sources = config["data"]["state_sources"]
    missing = [name for name in used if name not in sources]
    if missing:
        raise ValueError(f"data.mixing draws from pools without a data.state_sources entry: {missing}")
    restricted = copy.deepcopy(dict(config))
    restricted["data"] = dict(restricted["data"], state_sources={name: sources[name] for name in used})
    return restricted


# ------------------------------------------------------------------------------------------- one iteration
def train_iteration(train_model: SIRBTrain, optimizer: torch.optim.Optimizer, views: Sequence, *, clip_norm: float,
                    monitor: bool = True) -> dict:
    """One training iteration on ``views`` (module docstring). ``monitor=False`` reads nothing from the device but one
    finiteness check of all view losses; the gradients and the step are the same."""
    count = len(views)
    if count < 1:
        raise ValueError("a training iteration needs views")
    model = train_model.model
    optimizer.zero_grad(set_to_none=True)
    losses, readings, finite = [], [], []
    with train_model.shared_contexts() as shared:
        try:
            for view in views:
                loss, reading = shared.view_loss(view, monitor=monitor)
                if loss.ndim != 0:
                    raise FloatingPointError(NONFINITE_VIEW_LOSS)
                if monitor:
                    if not torch.isfinite(loss):
                        raise FloatingPointError(NONFINITE_VIEW_LOSS)
                else:
                    finite.append(torch.isfinite(loss.detach()))
                (loss / count).backward()
                if monitor:
                    losses.append(float(loss.detach()))
                    readings.append(plain_readings(reading))
                del loss, reading
            if finite and not bool(torch.stack(finite).all()):
                raise FloatingPointError(NONFINITE_VIEW_LOSS)
            shared.finish()
            branches = plain_readings(model.branch_gradient_norms()) if monitor else None
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm, error_if_nonfinite=True)
            optimizer.step()
        except BaseException:
            optimizer.zero_grad(set_to_none=True)
            raise
    if not monitor:
        return {"views": count, "monitored": False}
    return {"views": count, "monitored": True, "view_losses": losses, "loss": sum(losses) / count,
            "gradient_norm": float(norm), "branch_gradient_norms": branches, "view_readings": readings,
            "native_voxels": sum(reading["native_voxels"] for reading in readings)}


# ------------------------------------------------------------------------------------------- unit preparation
def _initialize_worker() -> None:
    torch.set_num_threads(1)


class UnitPreparer:
    """Prepares the units of reserved iterations (``sirb.data.pairs.prepare_unit`` with the source targets of the
    losses) in ``workers`` spawned processes, or in this process when ``workers`` is 0."""

    def __init__(self, *, workers: int) -> None:
        if isinstance(workers, bool) or not isinstance(workers, int) or workers < 0:
            raise ValueError(f"workers must be an integer >= 0, got {workers!r}")
        self.workers = workers
        self._executor: ProcessPoolExecutor | None = None

    def submit(self, context: UnitContext, iteration: int, requests: Mapping[str, Any]) -> list:
        units = [requests[kind] for kind in UNIT_ORDER]
        if not self.workers:
            return [prepare_unit(context, request, iteration, view_targets=source_targets) for request in units]
        if self._executor is None:
            self._executor = ProcessPoolExecutor(max_workers=self.workers,
                                                 mp_context=multiprocessing.get_context("spawn"),
                                                 initializer=_initialize_worker)
        return [self._executor.submit(prepare_unit, context, request, iteration, view_targets=source_targets)
                for request in units]

    def collect(self, ticket: list) -> list:
        return [task.result() for task in ticket] if self.workers else list(ticket)

    def discard(self, ticket: list) -> None:
        if self.workers:
            for task in ticket:
                task.cancel()

    def close(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=True)
            self._executor = None

    def __enter__(self) -> "UnitPreparer":
        return self

    def __exit__(self, *_) -> None:
        self.close()


@dataclass
class ReservedIteration:
    """One reserved iteration: its unit requests, its submitted preparation and the state of the streams and the
    course right after reserving it (what a checkpoint of this iteration stores)."""
    iteration: int
    requests: dict
    streams_state: dict
    course_state: dict
    ticket: list


class IterationPrefetch:
    """Reserve iterations i, i + 1, ... in order (at most ``depth`` not yet trained) and collect them in that order.

    :meth:`close` drops untrained reservations and rewinds the streams and the course to the last trained iteration,
    so a later reservation (or a resumed run) draws them again identically.
    """

    def __init__(self, sampler: TrainingPairSampler, preparer: UnitPreparer, context: UnitContext, *, depth: int):
        if isinstance(depth, bool) or not isinstance(depth, int) or depth < 1:
            raise ValueError(f"prefetch depth must be an integer >= 1, got {depth!r}")
        self.sampler, self.preparer, self.context, self.depth = sampler, preparer, context, depth
        self.pending: deque[ReservedIteration] = deque()
        self.trained_state = (copy.deepcopy(sampler.streams.state_dict()), copy.deepcopy(sampler.course.state_dict()))

    def _reserve(self, iteration: int) -> ReservedIteration:
        requests = self.sampler.reserve(iteration)
        streams_state = copy.deepcopy(self.sampler.streams.state_dict())
        course_state = copy.deepcopy(self.sampler.course.state_dict())
        ticket = self.preparer.submit(self.context, iteration, requests)
        return ReservedIteration(iteration, requests, streams_state, course_state, ticket)

    def fill(self, next_iteration: int, last_iteration: int, *, ahead: bool = True) -> None:
        """Reserve up to ``depth`` iterations (only the next one unless ``ahead``), never beyond ``last_iteration``."""
        if self.pending and self.pending[0].iteration != next_iteration:
            raise RuntimeError(f"prefetched iteration actual={self.pending[0].iteration} expected={next_iteration}")
        iteration = self.pending[-1].iteration + 1 if self.pending else next_iteration
        while len(self.pending) < (self.depth if ahead else 1) and iteration <= last_iteration:
            self.pending.append(self._reserve(iteration))
            iteration += 1

    def collect(self):
        """(oldest reservation, its prepared units, seconds spent waiting for them)."""
        reserved = self.pending.popleft()
        started = time.perf_counter()
        units = self.preparer.collect(reserved.ticket)
        return reserved, units, time.perf_counter() - started

    def trained(self, reserved: ReservedIteration) -> None:
        self.trained_state = (reserved.streams_state, reserved.course_state)

    def close(self) -> None:
        while self.pending:
            self.preparer.discard(self.pending.popleft().ticket)
        self.sampler.streams.load_state_dict(self.trained_state[0])
        self.sampler.course.load_state_dict(copy.deepcopy(self.trained_state[1]))


# ------------------------------------------------------------------------------------------- trainer
class Trainer:
    """Train SIRB on the configuration's training set (module docstring).

    ``roster`` ({examination: patient}) defaults to the training split list; ``resume`` is a checkpoint path, or
    ``True`` for the experiment's ``checkpoints/latest.pt``.
    """

    def __init__(self, config: Mapping[str, Any], *, device="cpu", roster: Mapping[str, str] | None = None,
                 resume: str | Path | bool | None = None, logger=None) -> None:
        self.config = copy.deepcopy(dict(config))
        self.settings = train_settings(self.config)
        self.model_settings = model_settings(self.config)
        self.points = [point for point in refresh_points(self.config) if point.iteration < self.settings["iterations"]]
        self.run_name = self.config["run"]
        self.folder = experiment_folder(self.config)
        self.device = torch.device(device)
        runtime = self.settings["runtime"]
        latest = self.folder / CHECKPOINTS / LATEST
        resume = None if resume is False else resume
        if resume is None and any((self.folder / CHECKPOINTS).glob("*.pt")):
            raise FileExistsError(f"{self.folder} holds checkpoints already: resume, or choose another run name")
        self.logger = logger or get_logger(f"sirb.training.{self.run_name}", log_file=self.folder / LOGS / "train.log")
        self.metrics = MetricsWriter(self.folder / LOGS / "train.jsonl")
        torch.set_num_threads(runtime["cpu_threads"])
        torch.backends.cuda.matmul.allow_tf32 = self.settings["tf32"]
        torch.backends.cudnn.allow_tf32 = self.settings["tf32"]
        torch.backends.cudnn.benchmark = runtime["cudnn_benchmark"]
        torch.manual_seed(self.settings["seed"])
        self.roster = dict(roster) if roster is not None else training_roster(self.config)
        model = build_sirb(self.config, device=self.device, activation_checkpointing=runtime["recompute_convolutions"],
                           source_chunk=runtime["source_chunk"], recompute_role_head=runtime["recompute_role_head"])
        loss = self.settings["loss"]
        self.train_model = SIRBTrain(model, grid_spacing_mm=self.model_settings["grid_spacing_mm"],
                                     whole_volume_shape=self.model_settings["whole_volume_shape"],
                                     loss_weights={key: loss[key] for key in LOSS_WEIGHT_KEYS},
                                     dice_epsilon=loss["dice_epsilon"],
                                     merged_non_target_roles=self.model_settings["merged_non_target_roles"],
                                     voxel_chunk=runtime["voxel_chunk"],
                                     recompute_native_voxels=runtime["recompute_native_voxels"])
        self.optimizer = make_optimizer(model.parameters(), self.settings["optimizer"])
        self.streams = RandomStreams(self.settings["seed"])
        self.iteration = 0
        self.self_generated: dict[str, Path] = {}
        self.seconds = {"train": 0.0, "refresh": 0.0}
        self.stop_requested = False
        self.preparer = UnitPreparer(workers=runtime["workers"])
        course_state = None
        if resume is not None:
            course_state = self._restore(latest if resume is True else Path(resume))
        save_config(self.config, self.folder / "config.json")
        self._open_pools(course_state)

    # ------------------------------------------------------------------ pools
    def pool_directory(self, name: str) -> Path:
        return self_generated_directory(self.config, self.run_name, name)

    def _open_pools(self, course_state: dict | None) -> None:
        """Read the pools in use (with the published self-generated pools) and build the sampler on the live streams;
        ``course_state`` continues the course."""
        rows, directories = read_training_pools(
            pools_in_use(self.config), roster=self.roster,
            self_generated={name: (directory, directory / POOL_MANIFEST)
                            for name, directory in self.self_generated.items()})
        self.sampler = TrainingPairSampler(self.config, rows, self.streams)
        if course_state is not None:
            self.sampler.course.load_state_dict(copy.deepcopy(course_state))
        self.context = UnitContext.from_config(self.config, cache_directory=cache_directory(self.config),
                                               pool_directories=directories, roster=self.roster,
                                               pet_reference=pet_reference_from_config(self.config))

    # ------------------------------------------------------------------ checkpoints
    def checkpoint(self, *, reserved: ReservedIteration | None = None, milestone: bool = False) -> Path:
        """Save ``checkpoints/latest.pt`` (and ``iteration_NNNNNN.pt`` for a milestone). Inside the loop the
        streams and the course are those recorded when the current iteration was reserved."""
        if reserved is not None and reserved.iteration != self.iteration:
            raise ValueError(f"checkpoint of iteration {self.iteration} with the reservation of {reserved.iteration}")
        streams = reserved.streams_state if reserved is not None else self.streams.state_dict()
        course = reserved.course_state if reserved is not None else self.sampler.course.state_dict()
        extra = {"optimizer": self.optimizer.state_dict(), "streams": copy.deepcopy(streams),
                 "course": copy.deepcopy(course), "self_generated": sorted(self.self_generated),
                 "seconds": dict(self.seconds), "torch_rng": torch.get_rng_state(),
                 "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}
        folder = self.folder / CHECKPOINTS
        path = save_checkpoint(folder / LATEST, self.train_model.model, config=self.config, iteration=self.iteration,
                               **extra)
        if milestone:
            save_checkpoint(folder / f"iteration_{self.iteration:06d}.pt", self.train_model.model, config=self.config,
                            iteration=self.iteration, **extra)
        return path

    def _restore(self, path: Path) -> dict:
        state = load_checkpoint(path, map_location="cpu")
        if state["config"] is None or recipe(state["config"]) != recipe(self.config):
            raise ValueError(f"{path} was trained with another configuration; only paths and train.runtime may "
                             f"change on resuming")
        self.train_model.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.streams.load_state_dict(state["streams"])
        self.iteration = int(state["iteration"])
        self.seconds.update(state["seconds"])
        torch.set_rng_state(state["torch_rng"])
        if torch.cuda.is_available() and len(state["cuda_rng"]) == torch.cuda.device_count():
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        for name in state["self_generated"]:
            directory = self.pool_directory(name)
            if not (directory / POOL_MANIFEST).is_file():
                raise FileNotFoundError(f"the self-generated pool {name} of this checkpoint is missing: {directory}")
            self.self_generated[name] = directory
        self.logger.info(f"resumed {self.run_name} at iteration {self.iteration} from {path}")
        return state["course"]

    # ------------------------------------------------------------------ training
    def _logged(self, iteration: int, boundary: int) -> bool:
        return iteration == 1 or iteration % self.settings["log_every"] == 0 or iteration == boundary

    def train_to(self, boundary: int) -> bool:
        """Train up to iteration ``boundary``; False when a stop was requested (after a checkpoint)."""
        settings, runtime = self.settings, self.settings["runtime"]
        cuda = self.device.type == "cuda"
        self.train_model.train()
        prefetch = IterationPrefetch(self.sampler, self.preparer, self.context, depth=runtime["prefetch_depth"])
        try:
            while self.iteration < boundary:
                started = time.perf_counter()
                prefetch.fill(self.iteration + 1, boundary, ahead=not self.stop_requested)
                reserved, units, waited = prefetch.collect()
                views = [view for unit in units for view in unit.views]
                if len(views) != settings["views_per_iteration"]:
                    raise ValueError(f"iteration {reserved.iteration} has {len(views)} views, expected "
                                     f"{settings['views_per_iteration']}")
                iteration = reserved.iteration
                rate = learning_rate(settings["learning_rate"], settings["optimizer"]["lr"], iteration)
                set_learning_rate(self.optimizer, rate)
                monitor = self._logged(iteration, boundary)
                if cuda and monitor:
                    torch.cuda.reset_peak_memory_stats(self.device)
                result = train_iteration(self.train_model, self.optimizer, views,
                                         clip_norm=settings["gradient_clip_norm"], monitor=monitor)
                if cuda:
                    torch.cuda.synchronize(self.device)
                elapsed = time.perf_counter() - started
                self.seconds["train"] += elapsed
                self.iteration = iteration
                prefetch.trained(reserved)
                if monitor:
                    row = {"iteration": iteration, "learning_rate": rate, "seconds": elapsed,
                           "data_wait_seconds": waited, "train_seconds": self.seconds["train"],
                           "pools": {kind: reserved.requests[kind].source for kind in UNIT_ORDER},
                           "modes": {kind: reserved.requests[kind].mode for kind in UNIT_ORDER},
                           "units": [unit.kind for unit in units], **result}
                    if cuda:
                        row["peak_allocated_gb"] = torch.cuda.max_memory_allocated(self.device) / 1024 ** 3
                    self.metrics.write(row)
                    self.logger.info(f"iteration {iteration}: loss {result['loss']:.4f}, learning rate {rate:.3g}, "
                                     f"{elapsed:.2f} s")
                if iteration % settings["checkpoint_every"] == 0 or iteration == boundary or self.stop_requested:
                    self.checkpoint(reserved=reserved, milestone=iteration == boundary)
                if self.stop_requested:
                    return False
        finally:
            prefetch.close()
        return True

    def refresh(self, points: Sequence[RefreshPoint]) -> None:
        """The self-generated states of ``points`` (all of this iteration), published together; then the sampler
        draws from them and a checkpoint records them."""
        started = time.perf_counter()
        self.optimizer.zero_grad(set_to_none=True)
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        model = self.train_model.model
        was_training = model.training
        generated = []
        try:
            for point in points:
                name = self_generated_pool_name(self.config, point)
                directory = self.pool_directory(name)
                self.logger.info(f"iteration {self.iteration}: self-generated states of patient half "
                                 f"{point.patient_half} in {directory}")
                rows = generate_pool(model, self.config, run=self.run_name, point=point, roster=self.roster,
                                     directory=directory, device=self.device, seed=self.settings["seed"],
                                     log=self.logger.info)
                generated.append((point, name, directory, rows))
        finally:
            model.train(was_training)
        for point, name, directory, rows in generated:
            publish_refresh(directory, rows, run=self.run_name, point=point, roster=self.roster)
            self.self_generated[name] = directory
        self._open_pools(self.sampler.course.state_dict())
        self.seconds["refresh"] += time.perf_counter() - started
        self.checkpoint()

    def _request_stop(self, *_) -> None:
        self.stop_requested = True
        self.logger.info("stop requested: finishing the current iteration and saving a checkpoint")

    def run(self, *, handle_signals: bool = False) -> dict:
        """Train to ``train.iterations``; returns {"iteration", "complete"}. With ``handle_signals`` SIGINT and SIGTERM
        stop the run after a checkpoint (resume with ``--resume``)."""
        if handle_signals:
            for name in ("SIGINT", "SIGTERM"):
                signal.signal(getattr(signal, name), self._request_stop)
        total = self.settings["iterations"]
        boundaries = sorted({point.iteration for point in self.points} | {total})
        try:
            for boundary in boundaries:
                if self.iteration < boundary and not self.train_to(boundary):
                    self.logger.info(f"stopped at iteration {self.iteration}")
                    return {"iteration": self.iteration, "complete": False}
                if self.iteration == boundary:
                    due = [point for point in self.points if point.iteration == boundary
                           and self_generated_pool_name(self.config, point) not in self.self_generated]
                    if due:
                        self.refresh(due)
        finally:
            self.preparer.close()
        write_json_atomic(self.folder / "complete.json",
                          {"run": self.run_name, "iteration": self.iteration, "seconds": self.seconds,
                           "self_generated": sorted(self.self_generated),
                           "checkpoint": f"{CHECKPOINTS}/iteration_{self.iteration:06d}.pt"})
        self.logger.info(f"finished {self.run_name} at iteration {self.iteration}")
        return {"iteration": self.iteration, "complete": True}
