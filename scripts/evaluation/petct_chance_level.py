#!/usr/bin/env python3
"""Non-model reference floors for the six-class joint-goal macro-F1.

The primary P2T number is ``patient_balanced_macro_f1`` over the six frozen
joint goals.  A bare score such as 0.76 is uninterpretable without the floor a
trivial predictor already reaches, so this module scores several *non-model*
policies with the exact same metric function the evaluator uses.

Every baseline is reported with:

* a D14 estimate obtained by drawing predictions and calling
  ``patient_balanced_macro_f1`` itself (the chance floor used for decisions),
  with a simulation interval; and
* a pooled ratio-of-expectations reference kept only as a diagnostic.

The pooled reference is not the D14 estimand: patient×class F1 is nonlinear,
undefined cells are skipped, and a class prior cannot encode the per-patient
class composition.  The two values must therefore never be substituted for
one another.

This module reads no image, mask, or patient record.  It needs only the gold
class distribution.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from common.petct_learning import patient_balanced_macro_f1  # noqa: E402
from common.petct_models import LEGAL_GOAL_SLOTS, LEGAL_GOALS  # noqa: E402


CLASS_ORDER = LEGAL_GOALS
CLASS_COUNT = len(CLASS_ORDER)
CLASS_LABELS = list(range(CLASS_COUNT))
# Slot 0 of the frozen joint definition is the operation id (0 = ADD, 1 = REMOVE).
OPERATION_OF_CLASS = tuple(slots[0] for slots in LEGAL_GOAL_SLOTS)
METRIC_REFERENCE = "scripts/common/petct_learning.py::patient_balanced_macro_f1"
DEFAULT_SEED = 20260806
DEFAULT_REPEATS = 1000

Policy = Tuple[Tuple[float, ...], ...]


class ChanceLevelError(ValueError):
    """Raised when the requested population or policy is not well formed."""


@dataclass(frozen=True)
class Population:
    """A gold-label population; predictions are drawn against it."""

    gold: Tuple[int, ...]
    patients: Tuple[str, ...]
    label: str = "population"

    @property
    def episode_count(self) -> int:
        return len(self.gold)

    @property
    def patient_count(self) -> int:
        return len(set(self.patients))

    def class_counts(self) -> Tuple[int, ...]:
        return tuple(
            sum(1 for value in self.gold if value == label) for label in CLASS_LABELS
        )


def build_population(
    class_counts: Optional[Sequence[int]],
    patient_composition: Optional[Mapping[str, Mapping[str, int]]] = None,
    label: str = "population",
) -> Population:
    """Build a gold population from class counts or a per-patient composition.

    ``class_counts`` alone cannot express patient structure, so each episode is
    placed in its own synthetic patient.  Patient-balanced weighting then
    degenerates to episode-level weighting; pass ``patient_composition`` when
    the real per-patient class mix is known and that distinction matters.
    """

    if (class_counts is None) == (patient_composition is None):
        raise ChanceLevelError(
            "supply exactly one of class_counts or patient_composition"
        )
    gold: List[int] = []
    patients: List[str] = []
    if class_counts is not None:
        if len(class_counts) != CLASS_COUNT:
            raise ChanceLevelError(
                "class_counts must hold %d values in LEGAL_GOALS order" % CLASS_COUNT
            )
        if any(int(count) < 0 for count in class_counts):
            raise ChanceLevelError("class_counts must be non-negative")
        for class_id, count in enumerate(class_counts):
            for _ in range(int(count)):
                gold.append(class_id)
                patients.append("synthetic-episode-%05d" % len(gold))
    else:
        for patient in sorted(patient_composition):
            counts = patient_composition[patient]
            unknown = sorted(set(counts) - set(CLASS_ORDER))
            if unknown:
                raise ChanceLevelError("unknown joint goal names: %s" % unknown)
            for class_id, name in enumerate(CLASS_ORDER):
                for _ in range(int(counts.get(name, 0))):
                    gold.append(class_id)
                    patients.append(str(patient))
    if not gold:
        raise ChanceLevelError("population is empty")
    return Population(tuple(gold), tuple(patients), label)


def weighted_prior(population: Population) -> Tuple[float, ...]:
    """Patient-balanced gold mass per class, normalised to sum to one."""

    episodes_per_patient: Dict[str, int] = {}
    for patient in population.patients:
        episodes_per_patient[patient] = episodes_per_patient.get(patient, 0) + 1
    mass = [0.0] * CLASS_COUNT
    for class_id, patient in zip(population.gold, population.patients):
        mass[class_id] += 1.0 / episodes_per_patient[patient]
    total = float(sum(mass))
    return tuple(value / total for value in mass)


def _validate_policy(policy: Policy) -> None:
    if len(policy) != CLASS_COUNT or any(len(row) != CLASS_COUNT for row in policy):
        raise ChanceLevelError(
            "policy must be a %dx%d matrix" % (CLASS_COUNT, CLASS_COUNT)
        )
    for row in policy:
        if any(value < 0.0 for value in row) or abs(sum(row) - 1.0) > 1e-9:
            raise ChanceLevelError("each policy row must be a probability vector")


def uniform_policy() -> Policy:
    """Guess one of the six classes with equal probability."""

    row = tuple(1.0 / CLASS_COUNT for _ in CLASS_LABELS)
    return tuple(row for _ in CLASS_LABELS)


def prior_matched_policy(prior: Sequence[float]) -> Policy:
    """Guess by sampling the gold class prior, ignoring the input."""

    row = tuple(float(value) for value in prior)
    return tuple(row for _ in CLASS_LABELS)


def constant_class_policy(class_id: int) -> Policy:
    """Always emit one fixed class."""

    if not 0 <= class_id < CLASS_COUNT:
        raise ChanceLevelError("class_id out of range")
    row = tuple(1.0 if index == class_id else 0.0 for index in CLASS_LABELS)
    return tuple(row for _ in CLASS_LABELS)


def majority_class_policy(prior: Sequence[float]) -> Tuple[int, Policy]:
    """Always emit the highest-mass class of the evaluated population."""

    class_id = max(CLASS_LABELS, key=lambda index: prior[index])
    return class_id, constant_class_policy(class_id)


def operation_oracle_policy() -> Policy:
    """Know the operation (ADD vs REMOVE), guess target and scope at random.

    ADD / REMOVE is readable straight off the signed scribble, so a predictor
    that gets it for free is not cheating.  This is the honest floor: whatever
    the model scores above this is what the learned part actually bought.
    """

    rows: List[Tuple[float, ...]] = []
    for gold in CLASS_LABELS:
        allowed = [
            index
            for index in CLASS_LABELS
            if OPERATION_OF_CLASS[index] == OPERATION_OF_CLASS[gold]
        ]
        share = 1.0 / len(allowed)
        rows.append(
            tuple(share if index in allowed else 0.0 for index in CLASS_LABELS)
        )
    return tuple(rows)


def is_deterministic(policy: Policy) -> bool:
    return all(any(value == 1.0 for value in row) for row in policy)


def analytic_macro_f1(prior: Sequence[float], policy: Policy) -> float:
    """Pooled asymptotic diagnostic; explicitly *not* the D14 chance floor.

    With weighted gold mass ``p`` and row-stochastic ``Q[gold][pred]`` the
    expected weighted confusion mass of class ``c`` is
    ``tp = p_c*Q[c][c]``, ``fp = sum_{g!=c} p_g*Q[g][c]``,
    ``fn = p_c*(1-Q[c][c])``. Undefined aggregate classes are skipped, as
    D14 skips undefined patient×class cells, but the remaining pooled ratios
    still cannot recover patient-level nonlinear averaging from a class prior.
    """

    _validate_policy(policy)
    if len(prior) != CLASS_COUNT:
        raise ChanceLevelError("prior must hold %d values" % CLASS_COUNT)
    scores = []
    for class_id in CLASS_LABELS:
        hit_rate = policy[class_id][class_id]
        tp = prior[class_id] * hit_rate
        fp = sum(
            prior[gold] * policy[gold][class_id]
            for gold in CLASS_LABELS
            if gold != class_id
        )
        fn = prior[class_id] * (1.0 - hit_rate)
        denom = 2.0 * tp + fp + fn
        if denom:
            scores.append(2.0 * tp / denom)
    if not scores:
        raise ChanceLevelError("pooled reference has no defined class")
    return float(np.mean(scores))


def make_rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(int(seed))


def sample_predictions(
    population: Population, policy: Policy, rng: np.random.Generator
) -> List[int]:
    """Draw one prediction per episode; identical seeds give identical output."""

    _validate_policy(policy)
    cumulative = np.cumsum(np.asarray(policy, dtype=np.float64), axis=1)
    gold = np.asarray(population.gold, dtype=np.int64)
    draws = rng.random(gold.size)
    chosen = (draws[:, None] >= cumulative[gold]).sum(axis=1)
    return np.minimum(chosen, CLASS_COUNT - 1).astype(int).tolist()


def monte_carlo_macro_f1(
    population: Population,
    policy: Policy,
    seed: int = DEFAULT_SEED,
    repeats: int = DEFAULT_REPEATS,
) -> Dict[str, object]:
    """Score a policy by repeatedly calling the real metric function."""

    if repeats < 1:
        raise ChanceLevelError("repeats must be >= 1")
    deterministic = is_deterministic(policy)
    effective = 1 if deterministic else int(repeats)
    rng = make_rng(seed)
    draws = [
        patient_balanced_macro_f1(
            population.gold,
            sample_predictions(population, policy, rng),
            population.patients,
            CLASS_LABELS,
        )
        for _ in range(effective)
    ]
    return {
        "mean": float(np.mean(draws)),
        "std": float(np.std(draws)),
        "ci_low": float(np.quantile(draws, 0.025)),
        "ci_high": float(np.quantile(draws, 0.975)),
        "repeats": effective,
        "seed": int(seed),
        "deterministic": deterministic,
    }


def usable_range_fraction(score: float, chance: float) -> float:
    """Where a score sits on ``[chance, 1.0]``; 0.0 at the floor, 1.0 at perfect."""

    if chance >= 1.0:
        raise ChanceLevelError("chance floor must be below 1.0")
    return (float(score) - float(chance)) / (1.0 - float(chance))


def _named_policies(prior: Sequence[float]) -> List[Tuple[str, str, Policy]]:
    majority_id, majority = majority_class_policy(prior)
    entries: List[Tuple[str, str, Policy]] = [
        ("uniform_random", "guess one of six classes with equal probability", uniform_policy()),
        (
            "prior_matched_random",
            "guess by sampling the gold class prior",
            prior_matched_policy(prior),
        ),
        (
            "majority_class",
            "always emit %s (largest weighted class)" % CLASS_ORDER[majority_id],
            majority,
        ),
        (
            "operation_oracle_random",
            "operation read off the scribble, target and scope random",
            operation_oracle_policy(),
        ),
    ]
    for class_id, name in enumerate(CLASS_ORDER):
        entries.append(
            (
                "constant_%s" % name,
                "always emit %s" % name,
                constant_class_policy(class_id),
            )
        )
    return entries


def baseline_table(
    population: Population,
    seed: int = DEFAULT_SEED,
    repeats: int = DEFAULT_REPEATS,
) -> List[Dict[str, object]]:
    prior = weighted_prior(population)
    rows: List[Dict[str, object]] = []
    for name, description, policy in _named_policies(prior):
        pooled_reference = analytic_macro_f1(prior, policy)
        simulated = monte_carlo_macro_f1(population, policy, seed=seed, repeats=repeats)
        rows.append(
            {
                "baseline": name,
                "description": description,
                "d14_chance_estimate": simulated["mean"],
                "d14_simulation_std": simulated["std"],
                "d14_simulation_ci_low": simulated["ci_low"],
                "d14_simulation_ci_high": simulated["ci_high"],
                "pooled_asymptotic_reference_not_d14": pooled_reference,
                "pooled_minus_d14_estimate": pooled_reference - float(simulated["mean"]),
                "repeats": simulated["repeats"],
                "deterministic": simulated["deterministic"],
                "usable_range": [simulated["mean"], 1.0],
            }
        )
    return rows


def zero_division_cost_probe() -> Dict[str, object]:
    """Record D14's supported-cell convention with the real metric."""

    present = build_population((1, 1, 1, 1, 1, 0))
    perfect_with_one_absent = patient_balanced_macro_f1(
        present.gold, present.gold, present.patients, CLASS_LABELS
    )
    full = build_population((1, 1, 1, 1, 1, 1))
    confused = list(full.gold)
    confused[5] = 4
    confusion_score = patient_balanced_macro_f1(
        full.gold, confused, full.patients, CLASS_LABELS
    )
    return {
        "rule": "patient×class cells with 2*tp+fp+fn == 0 are undefined and skipped; class support is reported",
        "cost_of_globally_absent_undefined_class": 0.0,
        "measured_perfect_score_with_one_class_absent": perfect_with_one_absent,
        "measured_cost_of_that_absent_class": 1.0 - perfect_with_one_absent,
        "measured_score_when_one_class_is_folded_into_another": confusion_score,
        "measured_cost_of_that_confusion": 1.0 - confusion_score,
        "note": (
            "A genuinely absent class is not imputed as perfect or zero. A real "
            "confusion remains defined through FN/FP cells and is penalized."
        ),
    }


def _parse_observed(values: Sequence[str]) -> List[Tuple[str, float]]:
    parsed: List[Tuple[str, float]] = []
    for raw in values:
        label, separator, number = raw.partition("=")
        text = number if separator else label
        try:
            parsed.append((label if separator else raw, float(text)))
        except ValueError:
            raise ChanceLevelError("observed score must be VALUE or LABEL=VALUE: %s" % raw)
    return parsed


def build_report(
    population: Population,
    observed: Sequence[Tuple[str, float]] = (),
    seed: int = DEFAULT_SEED,
    repeats: int = DEFAULT_REPEATS,
) -> Dict[str, object]:
    prior = weighted_prior(population)
    baselines = baseline_table(population, seed=seed, repeats=repeats)
    floors = {row["baseline"]: float(row["d14_chance_estimate"]) for row in baselines}
    observed_rows = [
        {
            "label": label,
            "score": float(score),
            "usable_range_fraction": {
                name: usable_range_fraction(score, floor)
                for name, floor in floors.items()
            },
        }
        for label, score in observed
    ]
    synthetic = len(set(population.patients)) == population.episode_count
    return {
        "metric": METRIC_REFERENCE,
        "class_order": list(CLASS_ORDER),
        "population": {
            "label": population.label,
            "episode_count": population.episode_count,
            "patient_count": population.patient_count,
            "class_counts": dict(zip(CLASS_ORDER, population.class_counts())),
            "weighted_prior": dict(zip(CLASS_ORDER, prior)),
            "weighting": (
                "episode-level (one synthetic patient per episode; supply "
                "--patient-composition for exact patient-balanced weighting)"
                if synthetic
                else "patient-balanced from the supplied per-patient composition"
            ),
        },
        "seed": int(seed),
        "repeats": int(repeats),
        "baselines": baselines,
        "chance_floor_rule": (
            "Use d14_chance_estimate/interval from the real patient-balanced scorer; "
            "pooled_asymptotic_reference_not_d14 is diagnostic only."
        ),
        "zero_division_convention": zero_division_cost_probe(),
        "observed": observed_rows,
    }


def _render_text(report: Mapping[str, object]) -> str:
    population = report["population"]
    lines = [
        "metric      : %s" % report["metric"],
        "population  : %s | %d episodes | %d patients"
        % (
            population["label"],
            population["episode_count"],
            population["patient_count"],
        ),
        "weighting   : %s" % population["weighting"],
        "seed/repeats: %s / %s" % (report["seed"], report["repeats"]),
        "",
        "%-38s %10s %10s %10s %8s" % ("baseline", "d14", "pooled", "sim_std", "det"),
        "-" * 80,
    ]
    for row in report["baselines"]:
        lines.append(
            "%-38s %10.4f %10.4f %10.4f %8s"
            % (
                row["baseline"],
                row["d14_chance_estimate"],
                row["pooled_asymptotic_reference_not_d14"],
                row["d14_simulation_std"],
                "yes" if row["deterministic"] else "no",
            )
        )
    if report["observed"]:
        floors = ["uniform_random", "prior_matched_random", "majority_class", "operation_oracle_random"]
        lines.extend(["", "share of the usable range [floor, 1.0]", "-" * 80])
        lines.append("%-24s %8s %s" % ("observed", "score", " ".join("%18s" % f for f in floors)))
        for row in report["observed"]:
            fractions = " ".join(
                "%17.1f%%" % (100.0 * row["usable_range_fraction"][name])
                for name in floors
            )
            lines.append("%-24s %8.4f %s" % (row["label"], row["score"], fractions))
    convention = report["zero_division_convention"]
    lines.extend(
        [
            "",
            "one globally absent undefined class costs %.4f macro-F1 (measured %.4f)"
            % (
                convention["cost_of_globally_absent_undefined_class"],
                convention["measured_cost_of_that_absent_class"],
            ),
        ]
    )
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--counts",
        type=int,
        nargs="+",
        help="six gold episode counts in LEGAL_GOALS order",
    )
    parser.add_argument(
        "--patient-composition",
        type=Path,
        help='JSON {"patient_id": {"ADD_SAME_LOCAL": 2, ...}} for exact patient weighting',
    )
    parser.add_argument("--label", default="population")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument(
        "--observed",
        nargs="*",
        default=[],
        help="measured scores to place on the usable range, as VALUE or LABEL=VALUE",
    )
    parser.add_argument("--format", choices=["text", "json"], default="text")
    args = parser.parse_args(argv)
    composition = None
    if args.patient_composition is not None:
        composition = json.loads(args.patient_composition.read_text(encoding="utf-8"))
    try:
        population = build_population(args.counts, composition, label=args.label)
        observed = _parse_observed(args.observed)
        report = build_report(
            population, observed, seed=args.seed, repeats=args.repeats
        )
    except ChanceLevelError as error:
        parser.error(str(error))
    if args.format == "json":
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(_render_text(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
