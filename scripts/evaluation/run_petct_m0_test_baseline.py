#!/usr/bin/env python3
"""Run the receipt-gated, test-only patient-excluded M0 baseline evaluation.

This runner deliberately reuses the committed ``OOF_READY`` predictions.  Each
case is scored from its one legal held-out-fold nnU-Net checkpoint; the runner
does not form a five-checkpoint ensemble and does not evaluate P2T or editor
quality.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import tempfile
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any


SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from common.petct_learning import (  # noqa: E402
    encode_json,
    encode_jsonl,
    load_jsonl,
    write_bytes_bundle_exclusive,
)
from common.petct_m0_test_access import enforce_m0_test_access  # noqa: E402
from data.build_petct_source_case_manifest import (  # noqa: E402
    LOCKED_STATE,
    MATERIALIZED_STATE,
    materialize_source_case_rows,
)
from data.validate_petct_learning_split import (  # noqa: E402
    load_and_validate_learning_split,
)
from evaluation.evaluate_petct_m0_oof import evaluate_m0_oof  # noqa: E402


PARTITION = "test"
OUTPUT_SOURCE_CASES = "source_cases.jsonl"
OUTPUT_METRIC_ROWS = "m0_test_rows.jsonl"
OUTPUT_SUMMARY = "m0_test_summary.json"
RUNNER_SCHEMA = "PETCT-M0-TEST-BASELINE-RUN-v1.0"
EXPECTED_EVALUATOR_SCHEMA = "PETCT-M0-OOF-EVALUATION-v1.1"
EXPECTED_EVALUATOR_STATUS = "COMPLETE_WITH_EXPLICIT_METRIC_ELIGIBILITY"
EXPECTED_TEST_CASES = 91
EXPECTED_LOCKED_TRAIN_VAL_CASES = 506
EXPECTED_HELD_OUT_FOLD_CASE_COUNTS = {0: 20, 1: 14, 2: 23, 3: 19, 4: 15}
EXPECTED_AGGREGATION_POPULATION = "positive_gt_eligible_cases_only"
EXPECTED_ELIGIBILITY_RULE = "GT contains at least one positive voxel"
EXPECTED_EMPTY_GT_POLICY = (
    "undefined for empty GT and serialized as JSON null; false positives "
    "remain explicit diagnostics"
)
EXPECTED_EVALUATOR_CLAIM_BOUNDARY = (
    "OOF M0 quality on explicitly selected frozen learning partitions only; "
    "not evidence that intent or correction works"
)
CLAIM_BOUNDARY = (
    "Locked test-partition M0 baseline from existing patient-excluded OOF_READY "
    "predictions: every case uses exactly one held-out-fold nnU-Net checkpoint. "
    "This is not a five-fold ensemble and is not evidence that P2T intent "
    "prediction or editor correction works."
)
EVALUATOR_SUMMARY_FIELDS = {
    "schema_version",
    "status",
    "selected_partitions",
    "test_access",
    "source_case_count",
    "case_count",
    "patient_count",
    "partition_case_counts",
    "oof_ready_sha256",
    "case_manifest_sha256",
    "truth_binding_sha256",
    "learning_split_sha256",
    "experiment_config_sha256",
    "official_metrics_sha256",
    "official_autoPETV",
    "positive_gt_patient_clustered",
    "empty_gt_false_positive_diagnostics",
    "claim_boundary",
}
PATIENT_CLUSTER_FIELDS = {
    "defined",
    "episode_count",
    "defined_episode_count",
    "patient_count",
    "mean",
    "median",
    "std",
    "std_defined",
}
OFFICIAL_METRIC_FIELDS = {
    "dsc",
    "dmm_f1_aggregated",
    "overlap_threshold",
    "connectivity",
    "aggregation_population",
    "eligibility_rule",
    "eligible_case_count",
    "eligible_patient_count",
    "ineligible_empty_gt_case_count",
    "denominators",
}
EMPTY_GT_FIELDS = {
    "case_count",
    "patient_count",
    "false_positive_case_count",
    "false_positive_patient_count",
    "false_positive_lesion_count",
    "prediction_volume_ml_total",
    "patient_clustered_fpv_ml",
    "patient_clustered_prediction_volume_ml",
    "official_dice_and_dmm_policy",
}
FORBIDDEN_PUBLIC_KEYS = {
    "case_id",
    "patient_id",
    "ct_path",
    "pet_path",
    "gt_path",
    "truth_binding_sha256",
    "case_manifest_sha256",
    "bound_run_root",
    "cases",
    "patients",
    "rows",
    "metric_rows",
}


def _load_json_object(path: Path) -> dict[str, Any]:
    raw = Path(path)
    if raw.is_symlink() or not raw.is_file():
        raise RuntimeError(f"missing regular frozen JSON: {raw}")
    try:
        payload = json.loads(raw.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid frozen JSON: {raw}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"frozen JSON must be an object: {raw}")
    return payload


def _receipt_sha256(receipt: Mapping[str, Any]) -> str:
    value = receipt.get("receipt_sha256")
    if not isinstance(value, str) or len(value) != 64:
        raise RuntimeError("M0 test-access receipt lacks receipt_sha256")
    try:
        int(value, 16)
    except ValueError as exc:
        raise RuntimeError("M0 test-access receipt_sha256 is not hexadecimal") from exc
    return value


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _finite_or_none(value: Any, *, label: str) -> float | None:
    if value is None:
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise RuntimeError(f"{label} must be finite numeric or null")
    return float(value)


def _copy_patient_cluster(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != PATIENT_CLUSTER_FIELDS:
        raise RuntimeError(f"{label} patient-cluster summary shape is invalid")
    if not isinstance(value.get("defined"), bool) or not isinstance(
        value.get("std_defined"), bool
    ):
        raise RuntimeError(f"{label} patient-cluster flags are invalid")
    copied = dict(value)
    for field in (
        "episode_count",
        "defined_episode_count",
        "patient_count",
        "mean",
        "median",
        "std",
    ):
        copied[field] = _finite_or_none(copied.get(field), label=f"{label}.{field}")
    return copied


def _copy_safe_aggregate_sections(
    metric_summary: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if set(metric_summary) != EVALUATOR_SUMMARY_FIELDS:
        raise RuntimeError("official evaluator summary field set is not pinned")

    official_raw = metric_summary.get("official_autoPETV")
    if not isinstance(official_raw, Mapping) or set(official_raw) != OFFICIAL_METRIC_FIELDS:
        raise RuntimeError("official AutoPET V aggregate shape is invalid")
    denominators = official_raw.get("denominators")
    if not isinstance(denominators, Mapping) or set(denominators) != {
        "dsc_cases",
        "dmm_tp",
        "dmm_fp",
        "dmm_fn",
        "dmm_gt_lesions",
    }:
        raise RuntimeError("official AutoPET V denominator shape is invalid")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in denominators.values()
    ):
        raise RuntimeError("official AutoPET V denominators are invalid")
    official = dict(official_raw)
    official["denominators"] = dict(denominators)
    if (
        official.get("overlap_threshold") != 0.1
        or official.get("connectivity") != 18
        or official.get("aggregation_population")
        != EXPECTED_AGGREGATION_POPULATION
        or official.get("eligibility_rule") != EXPECTED_ELIGIBILITY_RULE
        or isinstance(official.get("eligible_patient_count"), bool)
        or not isinstance(official.get("eligible_patient_count"), int)
        or not 0 < official["eligible_patient_count"] <= 57
    ):
        raise RuntimeError("official AutoPET V metadata contract is invalid")

    positive_raw = metric_summary.get("positive_gt_patient_clustered")
    if not isinstance(positive_raw, Mapping) or set(positive_raw) != {
        "dice",
        "dmm_f1",
        "fpv_ml",
        "fnv_ml",
    }:
        raise RuntimeError("positive-GT patient-cluster aggregate shape is invalid")
    positive = {
        metric: _copy_patient_cluster(value, label=f"positive_gt.{metric}")
        for metric, value in positive_raw.items()
    }

    empty_raw = metric_summary.get("empty_gt_false_positive_diagnostics")
    if not isinstance(empty_raw, Mapping) or set(empty_raw) != EMPTY_GT_FIELDS:
        raise RuntimeError("empty-GT aggregate shape is invalid")
    empty = dict(empty_raw)
    integer_empty_fields = (
        "case_count",
        "patient_count",
        "false_positive_case_count",
        "false_positive_patient_count",
        "false_positive_lesion_count",
    )
    if any(
        isinstance(empty.get(field), bool)
        or not isinstance(empty.get(field), int)
        or empty[field] < 0
        for field in integer_empty_fields
    ):
        raise RuntimeError("empty-GT count aggregate is invalid")
    prediction_volume = empty.get("prediction_volume_ml_total")
    if (
        isinstance(prediction_volume, bool)
        or not isinstance(prediction_volume, (int, float))
        or not math.isfinite(float(prediction_volume))
        or float(prediction_volume) < 0
        or empty.get("official_dice_and_dmm_policy") != EXPECTED_EMPTY_GT_POLICY
    ):
        raise RuntimeError("empty-GT metadata aggregate is invalid")
    empty["patient_clustered_fpv_ml"] = _copy_patient_cluster(
        empty_raw.get("patient_clustered_fpv_ml"), label="empty_gt.fpv_ml"
    )
    empty["patient_clustered_prediction_volume_ml"] = _copy_patient_cluster(
        empty_raw.get("patient_clustered_prediction_volume_ml"),
        label="empty_gt.prediction_volume_ml",
    )
    return official, positive, empty


def _assert_aggregate_only(value: Any, *, key: str = "root") -> None:
    if isinstance(value, Mapping):
        for child_key, child in value.items():
            if (
                child_key in FORBIDDEN_PUBLIC_KEYS
                or child_key.endswith("_path")
                or child_key.endswith("_paths")
            ):
                raise RuntimeError(f"aggregate summary contains forbidden key: {child_key}")
            _assert_aggregate_only(child, key=child_key)
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if key not in {"selected_partitions", "authorized_partitions"} or list(
            value
        ) != [PARTITION]:
            raise RuntimeError(f"aggregate summary contains a forbidden sequence in {key}")
        return
    if isinstance(value, str) and (
        value.startswith("/")
        or (len(value) >= 3 and value[1] == ":" and value[2] in {"\\", "/"})
    ):
        raise RuntimeError(f"aggregate summary contains an absolute path in {key}")


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _validate_materialization_boundary(
    rows: Sequence[Mapping[str, Any]],
    *,
    split_receipt: Mapping[str, Any],
) -> tuple[int, int]:
    expected_counts = split_receipt.get("case_counts")
    if not isinstance(expected_counts, Mapping):
        raise RuntimeError("validated learning split omits case_counts")
    try:
        counts = {name: int(expected_counts[name]) for name in ("train", "val", "test")}
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("validated learning split has invalid case_counts") from exc
    if any(count < 0 for count in counts.values()):
        raise RuntimeError("validated learning split has negative case_counts")
    if len(rows) != sum(counts.values()):
        raise RuntimeError(
            "materialized source inventory differs from frozen learning split"
        )

    materialized = 0
    locked = 0
    for raw in rows:
        partition = str(raw.get("partition") or "")
        state = raw.get("truth_materialization")
        if partition == PARTITION:
            if state != MATERIALIZED_STATE:
                raise RuntimeError("a frozen test row was not materialized")
            materialized += 1
            continue
        if partition not in {"train", "val"} or state != LOCKED_STATE:
            raise RuntimeError("train/val must remain LOCKED_UNREAD in the M0 test run")
        forbidden = {
            "nifti_shape",
            *(
                f"{modality}_{suffix}"
                for modality in ("ct", "pet", "gt")
                for suffix in ("bytes", "sha256")
            ),
        }
        if forbidden.intersection(raw):
            raise RuntimeError(
                "a locked train/val row exposes test-leaf materialization fields"
            )
        locked += 1

    if materialized != counts[PARTITION]:
        raise RuntimeError("materialized test count differs from frozen learning split")
    return materialized, locked


def run_m0_test_baseline(
    *,
    partition: str,
    oof_ready: Path,
    identity_manifest: Path,
    learning_split: Path,
    experiment_config: Path,
    official_metrics: Path,
    test_access_receipt: Path,
    run_root: Path,
    ledger_root: Path,
    access_enforcer: Callable[..., Mapping[str, Any]] = enforce_m0_test_access,
    identity_loader: Callable[[Path], list[dict[str, Any]]] = load_jsonl,
    experiment_loader: Callable[[Path], dict[str, Any]] = _load_json_object,
    split_loader: Callable[..., tuple[dict[str, Any], dict[str, Any]]] = (
        load_and_validate_learning_split
    ),
    materializer: Callable[..., tuple[list[dict[str, Any]], str]] = (
        materialize_source_case_rows
    ),
    evaluator: Callable[..., tuple[list[dict[str, Any]], dict[str, Any]]] = (
        evaluate_m0_oof
    ),
    bundle_writer: Callable[[Mapping[Path, bytes]], None] = (
        write_bytes_bundle_exclusive
    ),
) -> dict[str, Any]:
    """Publish one atomic, receipt-bound M0 test baseline artifact bundle."""

    if partition != PARTITION:
        raise RuntimeError("M0 test baseline requires exact partition 'test'")
    raw_run_root = Path(run_root)
    if raw_run_root.is_symlink():
        raise RuntimeError("run root must be a real non-symlink directory")
    resolved_run_root = raw_run_root.resolve()
    if not resolved_run_root.is_dir():
        raise RuntimeError("run root must already exist as a directory")

    summary_output = resolved_run_root / OUTPUT_SUMMARY
    output_paths = (summary_output,)
    if any(path.exists() or path.is_symlink() for path in output_paths):
        raise FileExistsError("refusing to overwrite an M0 test baseline output")

    runner_script = Path(__file__).resolve()
    evaluator_script = runner_script.with_name("evaluate_petct_m0_oof.py")

    # This is the first operation allowed to inspect the frozen test decision.
    # In particular, identity loading, split loading, NIfTI materialization and
    # evaluator invocation all happen only after this receipt revalidation.
    access_receipt = access_enforcer(
        receipt_path=Path(test_access_receipt),
        experiment_config=Path(experiment_config),
        learning_split=Path(learning_split),
        identity_manifest=Path(identity_manifest),
        oof_ready=Path(oof_ready),
        official_metrics=Path(official_metrics),
        evaluator_script=evaluator_script,
        runner_script=runner_script,
        run_root=resolved_run_root,
        output_paths=output_paths,
        ledger_root=Path(ledger_root),
    )
    if not isinstance(access_receipt, Mapping):
        raise RuntimeError("M0 test-access enforcer returned a non-object receipt")
    receipt_sha256 = _receipt_sha256(access_receipt)

    identity_rows = identity_loader(Path(identity_manifest).resolve())
    experiment = experiment_loader(Path(experiment_config).resolve())
    split_document, split_receipt = split_loader(
        Path(learning_split).resolve(), identity_rows, experiment
    )
    source_rows, _ = materializer(
        identity_rows,
        split_document,
        authorized_partitions=(PARTITION,),
    )
    materialized_count, locked_count = _validate_materialization_boundary(
        source_rows, split_receipt=split_receipt
    )
    if materialized_count != EXPECTED_TEST_CASES:
        raise RuntimeError("frozen M0 test partition must contain exactly 91 cases")
    if locked_count != EXPECTED_LOCKED_TRAIN_VAL_CASES:
        raise RuntimeError(
            "frozen train/val inventory must remain 506 LOCKED_UNREAD cases"
        )
    source_bytes = encode_jsonl(source_rows)
    source_sha256 = hashlib.sha256(source_bytes).hexdigest()

    # The official evaluator has an exclusive file interface.  Give it an
    # isolated directory under the receipt-bound root.  Source identities and
    # per-case metrics are deleted with it; only the aggregate summary crosses
    # the formal publication boundary after evaluation succeeds.
    with tempfile.TemporaryDirectory(
        prefix=".m0-test-baseline-eval-", dir=str(resolved_run_root)
    ) as temporary_name:
        temporary_root = Path(temporary_name)
        temporary_source = temporary_root / OUTPUT_SOURCE_CASES
        temporary_rows = temporary_root / OUTPUT_METRIC_ROWS
        temporary_summary = temporary_root / OUTPUT_SUMMARY
        temporary_source.write_bytes(source_bytes)

        def _already_enforced(selected: Sequence[str], **access: Any) -> dict[str, Any]:
            if tuple(selected) != (PARTITION,):
                raise RuntimeError("official evaluator attempted a non-test partition")
            expected_paths = {
                "receipt_path": Path(test_access_receipt),
                "experiment_config": Path(experiment_config),
                "learning_split": Path(learning_split),
            }
            if any(
                not isinstance(access.get(name), Path)
                or Path(access[name]).resolve() != expected.resolve()
                for name, expected in expected_paths.items()
            ):
                raise RuntimeError("official evaluator changed the M0 test receipt")
            if (
                not isinstance(access.get("run_root"), Path)
                or Path(access["run_root"]).resolve() != resolved_run_root
            ):
                raise RuntimeError(
                    "official evaluator changed the receipt-bound run root"
                )
            evaluated_outputs = access.get("output_paths")
            expected_evaluator_outputs = (
                temporary_rows.resolve(),
                temporary_summary.resolve(),
            )
            if (
                not isinstance(evaluated_outputs, Sequence)
                or isinstance(evaluated_outputs, (str, bytes))
                or tuple(Path(path).resolve() for path in evaluated_outputs)
                != expected_evaluator_outputs
            ):
                raise RuntimeError(
                    "official evaluator changed its exact ephemeral outputs"
                )
            return {**dict(access_receipt), "receipt_sha256": receipt_sha256}

        metric_rows, metric_summary = evaluator(
            oof_ready=Path(oof_ready),
            case_manifest=temporary_source,
            learning_split=Path(learning_split),
            experiment_config=Path(experiment_config),
            official_metrics=Path(official_metrics),
            rows_path=temporary_rows,
            summary_path=temporary_summary,
            partitions=(PARTITION,),
            test_access_receipt=Path(test_access_receipt),
            run_root=resolved_run_root,
            test_access_validator=_already_enforced,
        )

    if not isinstance(metric_summary, dict):
        raise RuntimeError("official evaluator returned a non-object summary")
    official, positive_clustered, empty_diagnostics = _copy_safe_aggregate_sections(
        metric_summary
    )
    if (
        metric_summary.get("schema_version") != EXPECTED_EVALUATOR_SCHEMA
        or metric_summary.get("status") != EXPECTED_EVALUATOR_STATUS
        or metric_summary.get("claim_boundary")
        != EXPECTED_EVALUATOR_CLAIM_BOUNDARY
    ):
        raise RuntimeError("official evaluator schema/status/claim boundary is invalid")
    for hash_field in (
        "oof_ready_sha256",
        "case_manifest_sha256",
        "learning_split_sha256",
        "experiment_config_sha256",
        "official_metrics_sha256",
    ):
        if not _is_sha256(metric_summary.get(hash_field)):
            raise RuntimeError(f"official evaluator {hash_field} is not SHA-256")
    if metric_summary.get("selected_partitions") != [PARTITION]:
        raise RuntimeError("official evaluator summary is not test-only")
    expected_summary_counts = {
        "source_case_count": 597,
        "case_count": EXPECTED_TEST_CASES,
        "patient_count": 57,
    }
    if any(
        metric_summary.get(field) != expected
        for field, expected in expected_summary_counts.items()
    ):
        raise RuntimeError("official evaluator summary count contract is invalid")
    if metric_summary.get("partition_case_counts") != {PARTITION: EXPECTED_TEST_CASES}:
        raise RuntimeError("official evaluator summary partition count is invalid")
    raw_test_access = metric_summary.get("test_access")
    if (
        not isinstance(raw_test_access, Mapping)
        or set(raw_test_access)
        != {"required", "consumed_receipt_sha256", "bound_run_root"}
        or raw_test_access.get("required") is not True
        or raw_test_access.get("consumed_receipt_sha256") != receipt_sha256
        or raw_test_access.get("bound_run_root") != str(resolved_run_root)
    ):
        raise RuntimeError("official evaluator test-access aggregate is invalid")
    eligible = official.get("eligible_case_count")
    empty = official.get("ineligible_empty_gt_case_count")
    if (
        isinstance(eligible, bool)
        or not isinstance(eligible, int)
        or eligible <= 0
        or isinstance(empty, bool)
        or not isinstance(empty, int)
        or empty < 0
        or eligible + empty != EXPECTED_TEST_CASES
    ):
        raise RuntimeError("official AutoPET V eligibility denominators are invalid")
    for metric_name in ("dsc", "dmm_f1_aggregated"):
        value = official.get(metric_name)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise RuntimeError(f"official AutoPET V {metric_name} is not finite")
        if not 0.0 <= float(value) <= 1.0:
            raise RuntimeError(f"official AutoPET V {metric_name} is outside [0, 1]")
    if any(str(row.get("partition") or "") != PARTITION for row in metric_rows):
        raise RuntimeError("official evaluator returned a non-test metric row")
    if len(metric_rows) != materialized_count:
        raise RuntimeError(
            "official test metric count differs from materialized test cases"
        )

    held_out_fold_counts: Counter[int] = Counter()
    for row in metric_rows:
        fold = row.get("held_out_fold")
        if isinstance(fold, bool) or not isinstance(fold, int) or fold not in range(5):
            raise RuntimeError("official evaluator returned an invalid held_out_fold")
        held_out_fold_counts[fold] += 1
    observed_fold_counts = {
        fold: int(held_out_fold_counts[fold]) for fold in range(5)
    }
    if observed_fold_counts != EXPECTED_HELD_OUT_FOLD_CASE_COUNTS:
        raise RuntimeError(
            "M0 test held-out-fold case counts differ from frozen inventory"
        )

    truth_binding_inventory = metric_summary.get("truth_binding_sha256")
    if not isinstance(truth_binding_inventory, Mapping) or len(
        truth_binding_inventory
    ) != EXPECTED_TEST_CASES:
        raise RuntimeError(
            "official evaluator summary lacks the complete truth binding inventory"
        )
    truth_binding_canonical_sha256 = _canonical_sha256(truth_binding_inventory)
    metric_rows_bytes = encode_jsonl(metric_rows)
    metric_rows_sha256 = hashlib.sha256(metric_rows_bytes).hexdigest()

    final_summary = {
        "schema_version": metric_summary.get("schema_version"),
        "status": metric_summary.get("status"),
        "selected_partitions": [PARTITION],
        "test_access": {
            "required": True,
            "stage": "M0_BASELINE_ONLY",
            "consumed_receipt_sha256": receipt_sha256,
        },
        **expected_summary_counts,
        "partition_case_counts": {PARTITION: EXPECTED_TEST_CASES},
        "oof_ready_sha256": metric_summary.get("oof_ready_sha256"),
        "learning_split_sha256": metric_summary.get("learning_split_sha256"),
        "experiment_config_sha256": metric_summary.get("experiment_config_sha256"),
        "official_metrics_sha256": metric_summary.get("official_metrics_sha256"),
        "official_autoPETV": official,
        "positive_gt_patient_clustered": positive_clustered,
        "empty_gt_false_positive_diagnostics": empty_diagnostics,
        "runner_schema_version": RUNNER_SCHEMA,
        "claim_boundary": CLAIM_BOUNDARY,
        "baseline_definition": {
            "prediction_source": "existing committed OOF_READY",
            "checkpoint_policy": (
                "exactly one patient-excluded held-out-fold checkpoint per case"
            ),
            "five_fold_ensemble": False,
            "p2t_evidence": False,
            "editor_evidence": False,
        },
        "source_materialization": {
            "authorized_partitions": [PARTITION],
            "materialized_test_case_count": materialized_count,
            "locked_unread_train_val_case_count": locked_count,
        },
        "ephemeral_source_manifest_sha256": source_sha256,
        "ephemeral_metric_rows_sha256": metric_rows_sha256,
        "truth_binding_inventory_canonical_sha256": (
            truth_binding_canonical_sha256
        ),
        "held_out_fold_case_counts": {
            str(fold): count for fold, count in observed_fold_counts.items()
        },
    }
    _assert_aggregate_only(final_summary)

    bundle_writer(
        {
            summary_output: encode_json(final_summary),
        }
    )
    return final_summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partition", choices=(PARTITION,), required=True)
    parser.add_argument("--oof-ready", type=Path, required=True)
    parser.add_argument("--identity-manifest", type=Path, required=True)
    parser.add_argument("--learning-split", type=Path, required=True)
    parser.add_argument("--experiment-config", type=Path, required=True)
    parser.add_argument("--official-metrics", type=Path, required=True)
    parser.add_argument("--test-access-receipt", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--ledger-root", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        summary = run_m0_test_baseline(
            partition=args.partition,
            oof_ready=args.oof_ready,
            identity_manifest=args.identity_manifest,
            learning_split=args.learning_split,
            experiment_config=args.experiment_config,
            official_metrics=args.official_metrics,
            test_access_receipt=args.test_access_receipt,
            run_root=args.run_root,
            ledger_root=args.ledger_root,
        )
    except (RuntimeError, FileExistsError) as exc:
        parser.error(str(exc))
    print(json.dumps(summary, allow_nan=False, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
