from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


PROJECT = Path(__file__).resolve().parents[1]
SCRIPTS = PROJECT / "scripts"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(SCRIPTS / "evaluation"))

import evaluate_petct_p2t  # noqa: E402
import petct_chance_level as chance  # noqa: E402
from common.petct_learning import patient_balanced_macro_f1  # noqa: E402


ONE_SIXTH = 1.0 / 6.0

# Frozen corpus counts in LEGAL_GOALS order, quoted from the signed-scribble
# corpus record.  They are inputs to the test, never tuned to an expected score.
VAL_COUNTS = (10, 14, 26, 21, 27, 27)
TRAIN_PLUS_VAL_COUNTS = (105, 74, 139, 124, 133, 164)


def _independent_analytic(prior, policy):
    """Closed form written independently of the implementation under test.

    For a predictor whose class choice depends only on the gold class through a
    row-stochastic matrix ``policy[gold][pred]``, the weighted confusion mass is
    tp_c = p_c * Q[c][c], fp_c = sum_{g != c} p_g * Q[g][c], fn_c = p_c * (1 - Q[c][c]).
    """

    scores = []
    for c in range(6):
        tp = prior[c] * policy[c][c]
        fp = sum(prior[g] * policy[g][c] for g in range(6) if g != c)
        fn = prior[c] * (1.0 - policy[c][c])
        denom = 2.0 * tp + fp + fn
        if denom:
            scores.append(2.0 * tp / denom)
    return sum(scores) / len(scores)


def test_module_reuses_the_exact_primary_metric_function() -> None:
    """The chance level must be scored by the same object the evaluator calls."""

    assert chance.patient_balanced_macro_f1 is patient_balanced_macro_f1
    assert (
        evaluate_petct_p2t.patient_balanced_macro_f1
        is chance.patient_balanced_macro_f1
    )


def test_class_order_and_operation_map_come_from_the_frozen_contract() -> None:
    assert chance.CLASS_ORDER == (
        "ADD_SAME_LOCAL",
        "REMOVE_SAME_LOCAL",
        "ADD_SAME_COMPLETE",
        "REMOVE_SAME_COMPLETE",
        "ADD_NEW_COMPLETE",
        "REMOVE_NEW_COMPLETE",
    )
    # 0 = ADD, 1 = REMOVE, taken from LEGAL_GOAL_SLOTS rather than re-typed.
    assert chance.OPERATION_OF_CLASS == (0, 1, 0, 1, 0, 1)


def test_perfect_prediction_scores_one_on_the_primary_metric() -> None:
    population = chance.build_population(VAL_COUNTS)
    assert patient_balanced_macro_f1(
        population.gold, population.gold, population.patients, list(range(6))
    ) == pytest.approx(1.0)


def test_a_class_absent_from_gold_and_prediction_is_undefined_and_skipped() -> None:
    """D14 does not invent a perfect or zero score for an unsupported class."""

    population = chance.build_population((1, 1, 1, 1, 1, 0))
    score = patient_balanced_macro_f1(
        population.gold, population.gold, population.patients, list(range(6))
    )
    assert score == pytest.approx(1.0)
    assert 1.0 - score == pytest.approx(0.0)


def test_zeroing_one_class_by_confusion_costs_more_than_one_sixth() -> None:
    """The 1/6 figure is the isolated term cost, not the cost of a real error."""

    population = chance.build_population((1, 1, 1, 1, 1, 1))
    predictions = list(population.gold)
    predictions[5] = 4  # every REMOVE_NEW_COMPLETE predicted as ADD_NEW_COMPLETE
    score = patient_balanced_macro_f1(
        population.gold, predictions, population.patients, list(range(6))
    )
    # With one synthetic patient per episode, class 4 is defined for the
    # correct and false-positive patients separately: mean (1 + 0) / 2.
    assert score == pytest.approx(0.75)
    assert 1.0 - score == pytest.approx(0.25)


def test_constant_class_baseline_distinguishes_d14_from_pooled_reference() -> None:
    population = chance.build_population((2, 1, 1, 0, 0, 0))
    policy = chance.constant_class_policy(0)
    prior = chance.weighted_prior(population)
    # Pooled diagnostic: class 0 F1=2/3 and classes 1,2 are zero; unsupported
    # classes are skipped, hence (2/3)/3=2/9. D14 instead averages patient F1
    # within each supported class and is exactly 1/6 for this population.
    assert chance.analytic_macro_f1(prior, policy) == pytest.approx(2.0 / 9.0)
    assert chance.monte_carlo_macro_f1(population, policy, seed=1, repeats=1)[
        "mean"
    ] == pytest.approx(ONE_SIXTH)


def test_prior_matched_random_is_exactly_one_sixth_for_any_prior() -> None:
    """2*p*q/(p+q) collapses to p when q == p, so the macro mean is sum(p)/6."""

    for counts in (VAL_COUNTS, TRAIN_PLUS_VAL_COUNTS, (1, 2, 3, 4, 5, 6)):
        prior = chance.weighted_prior(chance.build_population(counts))
        policy = chance.prior_matched_policy(prior)
        assert chance.analytic_macro_f1(prior, policy) == pytest.approx(
            ONE_SIXTH, abs=1e-12
        )


def test_no_class_independent_random_policy_can_beat_one_sixth() -> None:
    """Harmonic <= arithmetic mean bounds every unconditional guesser by 1/6."""

    prior = chance.weighted_prior(chance.build_population(TRAIN_PLUS_VAL_COUNTS))
    guesses = [
        (1 / 6,) * 6,
        (0.5, 0.1, 0.1, 0.1, 0.1, 0.1),
        (0.9, 0.02, 0.02, 0.02, 0.02, 0.02),
        tuple(prior),
    ]
    for q in guesses:
        policy = tuple(tuple(q) for _ in range(6))
        assert chance.analytic_macro_f1(prior, policy) <= ONE_SIXTH + 1e-12


def test_d14_deterministic_floor_is_scored_by_the_real_metric() -> None:
    """A class prior cannot reconstruct the nonlinear patient-level estimand."""

    composition = {
        "patient_a": {"ADD_SAME_LOCAL": 3, "REMOVE_NEW_COMPLETE": 1},
        "patient_b": {"ADD_SAME_COMPLETE": 1},
        "patient_c": {"REMOVE_SAME_COMPLETE": 2, "ADD_NEW_COMPLETE": 2},
        "patient_d": {"REMOVE_SAME_LOCAL": 5},
    }
    population = chance.build_population(None, patient_composition=composition)
    prior = chance.weighted_prior(population)
    observed_gap = False
    for class_id in range(6):
        policy = chance.constant_class_policy(class_id)
        measured = patient_balanced_macro_f1(
            population.gold,
            [class_id] * len(population.gold),
            population.patients,
            list(range(6)),
        )
        simulated = chance.monte_carlo_macro_f1(
            population, policy, seed=20260806, repeats=100
        )
        assert simulated["mean"] == pytest.approx(measured)
        observed_gap |= chance.analytic_macro_f1(prior, policy) != pytest.approx(
            measured
        )
    assert observed_gap


def test_weighted_prior_is_patient_balanced_not_episode_balanced() -> None:
    composition = {
        "patient_a": {"ADD_SAME_LOCAL": 3},
        "patient_b": {"REMOVE_NEW_COMPLETE": 1},
    }
    population = chance.build_population(None, patient_composition=composition)
    prior = chance.weighted_prior(population)
    assert prior[0] == pytest.approx(0.5)
    assert prior[5] == pytest.approx(0.5)


def test_monte_carlo_is_reproducible_under_a_fixed_seed() -> None:
    population = chance.build_population(VAL_COUNTS)
    policy = chance.uniform_policy()
    first = chance.sample_predictions(population, policy, chance.make_rng(20260806))
    second = chance.sample_predictions(population, policy, chance.make_rng(20260806))
    other = chance.sample_predictions(population, policy, chance.make_rng(20260807))
    assert first == second
    assert first != other
    left = chance.monte_carlo_macro_f1(population, policy, seed=20260806, repeats=25)
    right = chance.monte_carlo_macro_f1(population, policy, seed=20260806, repeats=25)
    assert left == right
    assert left["repeats"] == 25
    assert left["seed"] == 20260806


def test_uniform_d14_floor_is_reproducible_and_keeps_pooled_gap_explicit() -> None:
    population = chance.build_population(TRAIN_PLUS_VAL_COUNTS)
    prior = chance.weighted_prior(population)
    policy = chance.uniform_policy()
    analytic = chance.analytic_macro_f1(prior, policy)
    simulated = chance.monte_carlo_macro_f1(
        population, policy, seed=20260806, repeats=300
    )
    repeated = chance.monte_carlo_macro_f1(
        population, policy, seed=20260806, repeats=300
    )
    assert simulated == repeated
    assert simulated["ci_low"] <= simulated["mean"] <= simulated["ci_high"]
    # This is deliberate: the pooled ratio-of-expectations is only a
    # diagnostic and must not be substituted for the D14 chance floor.
    assert abs(float(simulated["mean"]) - analytic) > 0.01


def test_operation_oracle_uses_only_the_operation_and_beats_uniform() -> None:
    population = chance.build_population(TRAIN_PLUS_VAL_COUNTS)
    prior = chance.weighted_prior(population)
    policy = chance.operation_oracle_policy()
    for gold in range(6):
        for pred in range(6):
            same_operation = (
                chance.OPERATION_OF_CLASS[gold] == chance.OPERATION_OF_CLASS[pred]
            )
            assert policy[gold][pred] == pytest.approx(1.0 / 3.0 if same_operation else 0.0)
    oracle = chance.analytic_macro_f1(prior, policy)
    assert oracle == pytest.approx(_independent_analytic(prior, policy))
    assert oracle > chance.analytic_macro_f1(prior, chance.uniform_policy())
    assert oracle > ONE_SIXTH


def test_analytic_agrees_with_an_independently_written_closed_form() -> None:
    prior = chance.weighted_prior(chance.build_population(VAL_COUNTS))
    for policy in (
        chance.uniform_policy(),
        chance.prior_matched_policy(prior),
        chance.operation_oracle_policy(),
        chance.constant_class_policy(4),
    ):
        assert chance.analytic_macro_f1(prior, policy) == pytest.approx(
            _independent_analytic(prior, policy)
        )


def test_majority_class_policy_selects_the_largest_weighted_class() -> None:
    prior = chance.weighted_prior(chance.build_population(TRAIN_PLUS_VAL_COUNTS))
    class_id, policy = chance.majority_class_policy(prior)
    assert class_id == 5  # REMOVE_NEW_COMPLETE, 164 of 739
    assert policy == chance.constant_class_policy(5)


def test_usable_range_fraction_maps_chance_to_zero_and_one_to_one() -> None:
    assert chance.usable_range_fraction(0.3286, 0.3286) == pytest.approx(0.0)
    assert chance.usable_range_fraction(1.0, 0.3286) == pytest.approx(1.0)
    assert chance.usable_range_fraction(0.5, 0.0) == pytest.approx(0.5)
    # below the floor is reported honestly as a negative fraction
    assert chance.usable_range_fraction(0.2, 0.3286) < 0.0


def test_baseline_table_covers_every_required_reference_arm() -> None:
    population = chance.build_population(VAL_COUNTS)
    rows = chance.baseline_table(population, seed=20260806, repeats=20)
    names = [row["baseline"] for row in rows]
    assert "uniform_random" in names
    assert "prior_matched_random" in names
    assert "majority_class" in names
    assert "operation_oracle_random" in names
    assert sum(name.startswith("constant_") for name in names) == 6
    for row in rows:
        assert set(row) >= {
            "baseline",
            "d14_chance_estimate",
            "d14_simulation_std",
            "d14_simulation_ci_low",
            "d14_simulation_ci_high",
            "pooled_asymptotic_reference_not_d14",
            "pooled_minus_d14_estimate",
            "repeats",
            "deterministic",
        }
        assert row["d14_simulation_ci_low"] <= row["d14_chance_estimate"]
        assert row["d14_chance_estimate"] <= row["d14_simulation_ci_high"]
        assert row["usable_range"][0] == row["d14_chance_estimate"]


def test_cli_emits_a_json_report_with_ranges_for_observed_scores(capsys) -> None:
    exit_code = chance.main(
        [
            "--counts",
            "10",
            "14",
            "26",
            "21",
            "27",
            "27",
            "--label",
            "val",
            "--repeats",
            "20",
            "--seed",
            "20260806",
            "--observed",
            "primary=0.7616",
            "0.5795",
            "--format",
            "json",
        ]
    )
    assert exit_code == 0
    report = json.loads(capsys.readouterr().out)
    assert report["population"]["label"] == "val"
    assert report["population"]["episode_count"] == 125
    assert report["metric"].endswith("patient_balanced_macro_f1")
    assert "d14_chance_estimate" in report["chance_floor_rule"]
    # 4 named policies + one row per constant class
    assert len(report["baselines"]) == 10
    observed = {row["label"]: row for row in report["observed"]}
    assert set(observed) == {"primary", "0.5795"}
    assert observed["primary"]["score"] == pytest.approx(0.7616)
    floors = observed["primary"]["usable_range_fraction"]
    assert "uniform_random" in floors and "operation_oracle_random" in floors


def test_malformed_policy_is_rejected_with_a_readable_message() -> None:
    with pytest.raises(chance.ChanceLevelError, match="6x6 matrix"):
        chance.analytic_macro_f1((1 / 6,) * 6, ((1.0, 0.0),))
    with pytest.raises(chance.ChanceLevelError, match="probability vector"):
        chance.analytic_macro_f1((1 / 6,) * 6, ((0.5,) * 6,) * 6)


def test_population_requires_exactly_one_source_of_truth() -> None:
    with pytest.raises(chance.ChanceLevelError, match="exactly one"):
        chance.build_population(None, None)
    with pytest.raises(chance.ChanceLevelError, match="exactly one"):
        chance.build_population(VAL_COUNTS, {"p": {"ADD_SAME_LOCAL": 1}})
    with pytest.raises(chance.ChanceLevelError, match="unknown joint goal"):
        chance.build_population(None, {"p": {"ADD_NEW_LOCAL": 1}})


def test_cli_rejects_a_count_vector_that_is_not_six_classes() -> None:
    with pytest.raises(SystemExit):
        chance.main(["--counts", "1", "2", "3"])
