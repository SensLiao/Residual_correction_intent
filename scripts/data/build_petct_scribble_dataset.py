#!/usr/bin/env python3
"""Batch the pinned autoPET V simulator into auditable ADD/REMOVE cue episodes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Sequence
from uuid import uuid4

import nibabel as nib
import numpy as np
from scipy import ndimage

SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))
for support_dir in (SCRIPTS_ROOT / "baseline", SCRIPTS_ROOT / "data"):
    if str(support_dir) not in sys.path:
        sys.path.insert(0, str(support_dir))

from common.petct_learning import (  # noqa: E402
    LearningContractError,
    load_jsonl,
    sha256_file,
    validate_manifest_rows_against_frozen_learning_split,
)
from common.petct_test_access import (  # noqa: E402
    TestAccessError,
    add_leaf_test_access_arguments,
    enforce_partition_access,
)
from data.build_petct_scribble_episode import (  # noqa: E402
    AUTOPETV_RUNTIME_ALLOWLIST,
    AUTOPETV_RUNTIME_SCHEMA,
    AUTOPETV_RUNTIME_STATUS,
    CUE_ELIGIBILITY_RULE,
    CUE_INELIGIBLE_REASON,
    DEFAULT_RUNTIME_MANIFEST,
    DEFAULT_RUNTIME_MANIFEST_SHA256,
    STRATEGIES,
    ResidualCueIneligibleError,
    assign_scribble_strategy,
    build_episode_documents,
    canonical_intent_frame,
    compute_fn_residual,
    compute_fp_residual,
    generate_residual_scribble,
    load_official_simulator,
    publish_episode_documents,
)


LESION_CONNECTIVITY_18 = ndimage.generate_binary_structure(3, 2)
GENERATION_STAGE_ORDER = (
    "validated_m0_provenance",
    "full_fn_and_fp_residuals",
    "official_polarity_aware_residual_cue",
    "single_residual_component_binding",
    "canonical_intent_authorized_target",
)
GENERATION_CONTRACT_VERSION = "PETCT-CUE-GENERATION-v2.0"
RESIDUAL_READY_SCHEMA = "PETCT-FN-FP-RESIDUAL-READY-v2.0"
RESIDUAL_READY_PHASE = "FN_FP_RESIDUAL_DERIVATION"
SCRIBBLE_READY_SCHEMA = "PETCT-CUE-DATA-READY-v2.0"
SCRIBBLE_READY_PHASE = "OFFICIAL_RESIDUAL_CUE_EPISODE_MATERIALIZATION"


def _path_lexists(path: Path) -> bool:
    return os.path.lexists(str(path))


def _remove_owned_output(path: Path, *, directory: bool) -> None:
    if not _path_lexists(path):
        return
    if directory and path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink()


def _validate_output_layout(
    directory_outputs: Sequence[Path], file_outputs: Sequence[Path]
) -> None:
    tagged = [(Path(path).resolve(), True) for path in directory_outputs]
    tagged.extend((Path(path).resolve(), False) for path in file_outputs)
    for index, (first, first_is_dir) in enumerate(tagged):
        for second, second_is_dir in tagged[index + 1 :]:
            if first == second:
                raise RuntimeError("output paths must be distinct")
            nested = first in second.parents or second in first.parents
            if nested and (first_is_dir or second_is_dir):
                raise RuntimeError("file outputs must not be nested in output directories")


@contextmanager
def staged_output_bundle(
    *, directory_outputs: Sequence[Path], file_outputs: Sequence[Path]
) -> Iterator[dict[Path, Path]]:
    """Stage all episode outputs and publish the manifest only after validation."""

    directories = [Path(path).resolve() for path in directory_outputs]
    files = [Path(path).resolve() for path in file_outputs]
    _validate_output_layout(directories, files)
    tagged = [(path, True) for path in directories] + [(path, False) for path in files]
    for final, _ in tagged:
        final.parent.mkdir(parents=True, exist_ok=True)
        if _path_lexists(final):
            raise FileExistsError("refusing existing output path: %s" % final)
    staged = {
        final: final.with_name(
            ".%s.%d.%s.partial" % (final.name, os.getpid(), uuid4().hex)
        )
        for final, _ in tagged
    }
    if any(_path_lexists(stage) for stage in staged.values()):
        raise FileExistsError("generated staging path already exists")
    created: list[tuple[Path, bool]] = []
    try:
        for final, is_directory in tagged:
            if is_directory:
                staged[final].mkdir()
                created.append((staged[final], True))
    except Exception:
        for path, is_directory in reversed(created):
            _remove_owned_output(path, directory=is_directory)
        raise
    try:
        yield staged
    except Exception:
        for final, is_directory in tagged:
            _remove_owned_output(staged[final], directory=is_directory)
        raise
    committed: list[tuple[Path, bool]] = []
    try:
        for final, is_directory in tagged:
            if _path_lexists(final):
                raise FileExistsError("output appeared during staging: %s" % final)
            os.rename(staged[final], final)
            committed.append((final, is_directory))
    except Exception:
        for final, is_directory in tagged:
            _remove_owned_output(staged[final], directory=is_directory)
        for final, is_directory in reversed(committed):
            _remove_owned_output(final, directory=is_directory)
        raise


def _required_mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeError("%s must be an object" % label)
    return value


def _hex_digest(value: Any, *, length: int, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != length
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise RuntimeError("%s must be a lowercase hexadecimal digest" % label)
    return value


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def scribble_attempt_id(
    lane: str, case_id: str, operation: str, strategy: str
) -> str:
    if lane not in {"controlled", "natural", "controlled_p2t"}:
        raise ValueError("invalid scribble attempt lane")
    if strategy not in STRATEGIES:
        raise ValueError("invalid scribble attempt strategy")
    if operation not in {"ADD", "REMOVE"}:
        raise ValueError("invalid cue operation")
    return "scribble-attempt-" + hashlib.sha256(
        f"PETCT-CUE-ATTEMPT-v2|{lane}|{case_id}|{operation}|{strategy}".encode("utf-8")
    ).hexdigest()[:24]


def _validated_bucket(bucket: Any, *, label: str) -> dict[str, Any]:
    bucket = _required_mapping(bucket, label=label)
    case_ids = bucket.get("case_ids")
    patient_ids = bucket.get("patient_ids")
    if (
        not isinstance(case_ids, list)
        or not all(isinstance(item, str) and item for item in case_ids)
        or case_ids != sorted(set(case_ids))
    ):
        raise RuntimeError(f"{label}.case_ids must be sorted unique strings")
    if (
        not isinstance(patient_ids, list)
        or not all(isinstance(item, str) and item for item in patient_ids)
        or patient_ids != sorted(set(patient_ids))
    ):
        raise RuntimeError(f"{label}.patient_ids must be sorted unique strings")
    expected = {
        "case_count": len(case_ids),
        "patient_count": len(patient_ids),
        "case_ids_sha256": _canonical_hash(case_ids),
        "patient_ids_sha256": _canonical_hash(patient_ids),
    }
    for key, value in expected.items():
        if bucket.get(key) != value:
            raise RuntimeError(f"{label}.{key} mismatch")
    return dict(bucket)


def _validate_bound_file_record(
    record: Any, *, expected_path: Path, label: str
) -> dict[str, Any]:
    record = _required_mapping(record, label=label)
    raw_path = Path(str(record.get("path") or ""))
    if raw_path.is_symlink():
        raise RuntimeError(f"{label} path must not be a symlink")
    path = raw_path.resolve()
    if path != expected_path.resolve() or path.is_symlink() or not path.is_file():
        raise RuntimeError(f"{label} path binding mismatch")
    digest = sha256_file(path)
    if record.get("sha256") != digest or record.get("bytes") != path.stat().st_size:
        raise RuntimeError(f"{label} file binding mismatch")
    return dict(record)


def validate_residual_ready(
    ready_path: Path,
    *,
    residual_manifest: Path,
    oof_ready: Path,
    selected_partitions: set[str],
) -> dict[str, Any]:
    """Validate the hash-bound residual denominator without reading image leaves."""

    raw = Path(ready_path)
    if raw.is_symlink():
        raise RuntimeError("RESIDUAL_READY must be a regular non-symlink file")
    path = raw.resolve()
    if not path.is_file():
        raise RuntimeError("RESIDUAL_READY is missing")
    try:
        ready = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("RESIDUAL_READY is invalid UTF-8 JSON") from exc
    ready = _required_mapping(ready, label="RESIDUAL_READY")
    expected_header = {
        "schema_version": RESIDUAL_READY_SCHEMA,
        "status": "PASS",
        "phase": RESIDUAL_READY_PHASE,
    }
    for key, value in expected_header.items():
        if ready.get(key) != value:
            raise RuntimeError(f"RESIDUAL_READY {key} mismatch")
    if set(ready.get("selected_partitions", [])) != selected_partitions:
        raise RuntimeError("RESIDUAL_READY selected partitions mismatch")
    _validate_bound_file_record(
        ready.get("residual_manifest"),
        expected_path=residual_manifest,
        label="RESIDUAL_READY residual_manifest",
    )
    _validate_bound_file_record(
        ready.get("oof_ready"),
        expected_path=oof_ready,
        label="RESIDUAL_READY oof_ready",
    )
    cohort = _required_mapping(ready.get("cohort"), label="RESIDUAL_READY cohort")
    buckets = {
        key: _validated_bucket(cohort.get(key), label=f"RESIDUAL_READY cohort.{key}")
        for key in (
            "source",
            "selected_source",
            "generated",
            "fn_positive",
            "zero_fn",
            "fp_positive",
            "zero_fp",
            "excluded",
        )
    }
    selected_cases = set(buckets["selected_source"]["case_ids"])
    generated_cases = set(buckets["generated"]["case_ids"])
    positive_cases = set(buckets["fn_positive"]["case_ids"])
    zero_cases = set(buckets["zero_fn"]["case_ids"])
    fp_positive_cases = set(buckets["fp_positive"]["case_ids"])
    zero_fp_cases = set(buckets["zero_fp"]["case_ids"])
    if generated_cases != selected_cases:
        raise RuntimeError("RESIDUAL_READY generated denominator differs from selected source")
    if positive_cases & zero_cases or positive_cases | zero_cases != generated_cases:
        raise RuntimeError("RESIDUAL_READY FN-positive/zero-FN decomposition mismatch")
    if fp_positive_cases & zero_fp_cases or fp_positive_cases | zero_fp_cases != generated_cases:
        raise RuntimeError("RESIDUAL_READY FP-positive/zero-FP decomposition mismatch")
    return {
        **dict(ready),
        "ready_path": str(path),
        "ready_sha256": sha256_file(path),
        "validated_cohort": buckets,
    }


def _tree_record(stage_root: Path, final_root: Path) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    for path in sorted(stage_root.rglob("*")):
        if path.is_symlink():
            raise RuntimeError("staged output tree contains a symlink")
        if path.is_dir():
            continue
        if not path.is_file():
            raise RuntimeError("staged output tree contains a non-regular entry")
        entries.append(
            {
                "path": path.relative_to(stage_root).as_posix(),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
        )
    return {
        "path": str(final_root),
        "file_count": len(entries),
        "bytes": sum(item["bytes"] for item in entries),
        "tree_sha256": _canonical_hash(entries),
    }


def _output_file_record(stage_path: Path, final_path: Path) -> dict[str, Any]:
    return {
        "path": str(final_path),
        "bytes": stage_path.stat().st_size,
        "sha256": sha256_file(stage_path),
    }


def _cohort_bucket(case_ids: set[str], patient_ids: set[str]) -> dict[str, Any]:
    cases = sorted(case_ids)
    patients = sorted(patient_ids)
    return {
        "case_count": len(cases),
        "patient_count": len(patients),
        "case_ids": cases,
        "patient_ids": patients,
        "case_ids_sha256": _canonical_hash(cases),
        "patient_ids_sha256": _canonical_hash(patients),
    }


def _exclusion_summary(exclusions: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    reasons: dict[str, list[str]] = {}
    for exclusion in exclusions:
        reason = str(exclusion["reason"])
        reasons.setdefault(reason, []).append(str(exclusion["attempt_id"]))
    return {
        reason: {
            "count": len(set(attempt_ids)),
            "attempt_ids": sorted(set(attempt_ids)),
            "attempt_ids_sha256": _canonical_hash(sorted(set(attempt_ids))),
        }
        for reason, attempt_ids in sorted(reasons.items())
    }


def resolve_scribble_generation_contract(
    experiment_config: Mapping[str, Any],
    *,
    official_commit: str | None = None,
    strategy_mode: str | None = None,
    strategy_salt: str | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    """Resolve CLI overrides against the one frozen experiment configuration."""

    config = _required_mapping(experiment_config, label="experiment config")
    scribble = _required_mapping(config.get("scribble"), label="scribble config")
    if scribble.get("source") != "lab-midas/autoPETV":
        raise RuntimeError("scribble source must be lab-midas/autoPETV")
    configured_commit = _hex_digest(
        scribble.get("commit"), length=40, label="scribble.commit"
    )
    simulator_sha256 = _hex_digest(
        scribble.get("simulator_file_sha256"),
        length=64,
        label="scribble.simulator_file_sha256",
    )
    if official_commit is not None and official_commit != configured_commit:
        raise RuntimeError("official commit differs from experiment config")
    polarity_contract = _required_mapping(
        scribble.get("polarity_contract"), label="scribble polarity contract"
    )
    expected_polarity_contract = {
        "ADD": {
            "name": "POSITIVE_FOREGROUND",
            "source_residual": "FN",
            "signed_value": 1,
        },
        "REMOVE": {
            "name": "NEGATIVE_BACKGROUND",
            "source_residual": "FP",
            "signed_value": -1,
        },
    }
    if polarity_contract != expected_polarity_contract:
        raise RuntimeError(
            "scribble polarity contract must bind ADD->positive FN and "
            "REMOVE->negative FP"
        )
    model_encoding = _required_mapping(
        scribble.get("model_encoding"), label="scribble model encoding"
    )
    if model_encoding != {
        "p2t": "two disjoint binary channels: cue_fg and cue_bg",
        "editor_projection": (
            "one signed channel cue_fg-cue_bg: +1 ADD, -1 REMOVE, 0 elsewhere"
        ),
    }:
        raise RuntimeError("scribble model encoding differs from the v2 contract")
    if scribble.get("strategies") != list(STRATEGIES):
        raise RuntimeError("configured scribble strategies/order differ from official strategies")
    if scribble.get("primary_assignment") != "stable-patient-hash":
        raise RuntimeError("primary strategy assignment must be stable-patient-hash")
    if scribble.get("primary_strategy_mode") != "primary":
        raise RuntimeError("configured primary strategy mode must be primary")
    if scribble.get("robustness_strategy_mode") != (
        "all-as-three-matched-attempts-not-three-strokes-in-one-episode"
    ):
        raise RuntimeError("configured robustness strategy mode differs from v2")
    configured_salt = scribble.get("primary_strategy_salt")
    if not isinstance(configured_salt, str) or not configured_salt:
        raise RuntimeError("configured primary strategy salt is required")
    if strategy_salt is not None and strategy_salt != configured_salt:
        raise RuntimeError("strategy salt differs from experiment config")
    configured_seed = scribble.get("seed")
    if isinstance(configured_seed, bool) or not isinstance(configured_seed, int):
        raise RuntimeError("configured scribble seed must be an integer")
    if seed is not None and seed != configured_seed:
        raise RuntimeError("scribble seed differs from experiment config")
    configured_minimum_pixels = scribble.get("minimum_best_slice_pixels")
    if (
        isinstance(configured_minimum_pixels, bool)
        or not isinstance(configured_minimum_pixels, int)
        or configured_minimum_pixels < 1
    ):
        raise RuntimeError(
            "configured scribble minimum_best_slice_pixels must be a positive integer"
        )
    # Pinning the rule string keeps the unit auditable: a later edit to a mm^2
    # threshold cannot silently replace the per-case-spacing-safe pixel rule.
    if scribble.get("cue_eligibility_rule") != CUE_ELIGIBILITY_RULE:
        raise RuntimeError(
            "configured cue eligibility rule must be measured in pixels per "
            "residual component"
        )
    resolved_mode = strategy_mode or str(scribble["primary_strategy_mode"])
    robustness_mode = "all"
    allowed_modes = {str(scribble["primary_strategy_mode"]), robustness_mode}
    if resolved_mode not in allowed_modes:
        raise RuntimeError("strategy mode differs from experiment config")
    return {
        "contract_version": GENERATION_CONTRACT_VERSION,
        "source": str(scribble["source"]),
        "official_commit": configured_commit,
        "simulator_file_sha256": simulator_sha256,
        "polarity_contract": expected_polarity_contract,
        "model_encoding": dict(model_encoding),
        "strategies": list(scribble["strategies"]),
        "strategy_assignment": str(scribble["primary_assignment"]),
        "strategy_assignment_unit": "patient",
        "strategy_mode": resolved_mode,
        "robustness_strategy_contract": str(scribble["robustness_strategy_mode"]),
        "strategy_salt": configured_salt,
        "seed": int(configured_seed),
        "cue_eligibility_rule": CUE_ELIGIBILITY_RULE,
        "minimum_best_slice_pixels": int(configured_minimum_pixels),
        "stage_order": list(GENERATION_STAGE_ORDER),
    }


def validate_official_simulator_provenance(
    provenance: Mapping[str, Any], generation: Mapping[str, Any]
) -> None:
    provenance = _required_mapping(provenance, label="official simulator provenance")
    expected = {
        "repository": generation["source"],
        "commit": generation["official_commit"],
        "file_sha256": generation["simulator_file_sha256"],
    }
    for key, value in expected.items():
        if provenance.get(key) != value:
            raise RuntimeError("official simulator provenance differs for %s" % key)
    relative_path = provenance.get("relative_path")
    if relative_path != "interactive/simulate_scribbles.py":
        raise RuntimeError("official simulator provenance relative_path mismatch")
    mode = provenance.get("provenance_mode")
    if mode is None and provenance.get("git_worktree") == "CLEAN_FOR_SIMULATOR_FILE":
        # Backwards-compatible fixture/local receipt produced before the
        # minimal-runtime provenance field was introduced.
        mode = "CLEAN_GIT_WORKTREE"
    if mode == "CLEAN_GIT_WORKTREE":
        if provenance.get("git_worktree") != "CLEAN_FOR_SIMULATOR_FILE":
            raise RuntimeError("official simulator Git worktree is not clean")
        return
    if mode != "FROZEN_MINIMAL_RUNTIME":
        raise RuntimeError("official simulator provenance mode is unsupported")
    manifest = _required_mapping(
        provenance.get("runtime_manifest"), label="official runtime manifest provenance"
    )
    if (
        manifest.get("schema_version") != AUTOPETV_RUNTIME_SCHEMA
        or manifest.get("status") != AUTOPETV_RUNTIME_STATUS
        or manifest.get("sha256") != DEFAULT_RUNTIME_MANIFEST_SHA256
    ):
        raise RuntimeError("official minimal runtime manifest provenance mismatch")
    _hex_digest(
        provenance.get("runtime_bundle_sha256"),
        length=64,
        label="official runtime bundle sha256",
    )
    files = provenance.get("runtime_files")
    if (
        not isinstance(files, list)
        or sorted(str(item.get("path")) for item in files if isinstance(item, Mapping))
        != list(AUTOPETV_RUNTIME_ALLOWLIST)
    ):
        raise RuntimeError("official minimal runtime inventory provenance mismatch")


def selected_strategies(patient_id: str, mode: str, salt: str) -> List[str]:
    if mode == "all":
        return list(STRATEGIES)
    if mode == "primary":
        return [assign_scribble_strategy(patient_id, salt=salt)]
    raise ValueError("strategy mode must be primary or all")


def apply_strategy_identity_policy(
    record: Mapping[str, Any], *, strategy_mode: str, context: str
) -> dict[str, str] | None:
    """Enforce that a requested comparison arm is never silently relabelled."""

    requested = record.get("requested_strategy")
    effective = record.get("effective_strategy")
    fallback = record.get("strategy_fallback")
    if requested not in STRATEGIES or effective not in STRATEGIES:
        raise RuntimeError(f"{context} has invalid requested/effective strategy")
    if not isinstance(fallback, bool) or fallback != (requested != effective):
        raise RuntimeError(f"{context} has inconsistent strategy fallback audit")
    if not fallback:
        return None
    detail = str(record.get("fallback_reason") or "UNSPECIFIED_UPSTREAM_FALLBACK")
    if strategy_mode == "primary":
        raise RuntimeError(
            f"{context} crossed strategy arms and failed closed: "
            f"{requested}->{effective} ({detail})"
        )
    if strategy_mode != "all":
        raise RuntimeError("invalid strategy mode")
    return {
        "effective_strategy": str(effective),
        "reason": "CROSS_STRATEGY_FALLBACK",
        "detail": detail,
    }


def opaque_episode_id(case_id: str, goal: str, strategy: str) -> str:
    digest = hashlib.sha256(
        ("PETCT-EPISODE-v2|%s|%s|%s" % (case_id, goal, strategy)).encode("utf-8")
    ).hexdigest()
    return "petct-" + digest[:24]


def verify_full_residual(
    gt: np.ndarray,
    m0: np.ndarray,
    observed_residual: np.ndarray,
    *,
    operation: str,
) -> np.ndarray:
    if operation == "ADD":
        expected = compute_fn_residual(gt, m0)
    elif operation == "REMOVE":
        expected = compute_fp_residual(gt, m0)
    else:
        raise RuntimeError("operation must be ADD or REMOVE")
    observed = np.asarray(observed_residual) > 0
    if expected.shape != observed.shape or not np.array_equal(expected, observed):
        formula = "FN=GT\\M0" if operation == "ADD" else "FP=M0\\GT"
        raise RuntimeError(f"mining residual is not exactly {formula}")
    return observed


# Eight structurally different failures used to collapse into the single
# exclusion reason "STATE_RELATIVE_TARGET_INELIGIBLE", which made a genuine
# internal-consistency violation indistinguishable from an episode that is
# simply not eligible.  Four of them are legitimate ineligibility and keep a
# reason of their own; the other four mean the inputs or the derivation itself
# are broken and must stop the build rather than shrink the corpus quietly.
DERIVATION_REFUSAL_REASONS = {
    "cue must bind exactly one source component": "CUE_BINDS_MULTIPLE_COMPONENTS",
    "cued component has no operation-specific residual": "CUED_COMPONENT_HAS_NO_RESIDUAL",
    "current protocol requires a one-slice scribble": "SCRIBBLE_SPANS_MULTIPLE_SLICES",
    "SAME_LOCAL candidate is below minimum physical area": "LOCAL_BELOW_MINIMUM_AREA",
}
DERIVATION_HARD_FAILURES = (
    "GT/M0 must be aligned 3D masks",
    "operation must be ADD or REMOVE",
    "positive in-plane spacing is required",
    # The derivation contradicting itself is never an "ineligible episode".
    "derived authorized target does not contain the scribble",
)


def classify_derivation_refusal(error: RuntimeError) -> str:
    """Name the specific refusal, or re-raise when the failure is not one.

    Silently excluding a broken derivation shrinks the corpus and hides the
    break; the audit called this the single finding most directly endangering
    the six-class feasibility freeze.
    """

    message = str(error)
    for fragment in DERIVATION_HARD_FAILURES:
        if fragment in message:
            raise RuntimeError(
                "gold derivation violated its own contract, refusing to record "
                f"this as an ineligible episode: {message}"
            ) from error
    for fragment, reason in DERIVATION_REFUSAL_REASONS.items():
        if fragment in message:
            return reason
    raise RuntimeError(
        f"unclassified gold-derivation refusal, refusing to swallow it: {message}"
    ) from error


def derive_goal_and_authorized_target(
    *,
    gt: np.ndarray,
    m0: np.ndarray,
    operation: str,
    coordinates_xyz: List[List[int]],
    spacing_xy,
    local_radius_mm: float = 15.0,
    minimum_local_area_mm2: float = 50.0,
):
    """Bind one polarity-aware cue to one FN/FP component.

    ADD matches the cued FN to an 18-connected GT lesion. REMOVE matches the
    cued FP to an 18-connected current-mask component. SAME means that the
    matched source component also contains a retained counterpart (M0 for ADD,
    GT for REMOVE). NEW_LOCAL is never emitted.
    """

    truth = np.asarray(gt) > 0
    current = np.asarray(m0) > 0
    if truth.shape != current.shape or truth.ndim != 3:
        raise RuntimeError("GT/M0 must be aligned 3D masks")
    if operation == "ADD":
        source_mask = truth
        residual_mask = truth & ~current
        counterpart = current
    elif operation == "REMOVE":
        source_mask = current
        residual_mask = current & ~truth
        counterpart = truth
    else:
        raise RuntimeError("operation must be ADD or REMOVE")
    labels, _ = ndimage.label(source_mask, structure=LESION_CONNECTIVITY_18)
    scribble_labels = {int(labels[tuple(int(v) for v in coord)]) for coord in coordinates_xyz}
    if 0 in scribble_labels or len(scribble_labels) != 1:
        raise RuntimeError("cue must bind exactly one source component")
    target_component = labels == next(iter(scribble_labels))
    target_residual = target_component & residual_mask
    if not target_residual.any():
        raise RuntimeError("cued component has no operation-specific residual")
    slices = {int(coord[2]) for coord in coordinates_xyz}
    if len(slices) != 1:
        raise RuntimeError("current protocol requires a one-slice scribble")
    center_z = next(iter(slices))
    scribble_2d = np.zeros(truth.shape[:2], dtype=bool)
    for x, y, _ in coordinates_xyz:
        scribble_2d[int(x), int(y)] = True
    spacing = tuple(float(value) for value in spacing_xy)
    if len(spacing) != 2 or any(value <= 0 for value in spacing):
        raise RuntimeError("positive in-plane spacing is required")
    distance = ndimage.distance_transform_edt(~scribble_2d, sampling=spacing)
    residual_slice = target_residual[:, :, center_z]
    local_slice = residual_slice & (distance <= float(local_radius_mm))
    far_slice = residual_slice & (distance > float(local_radius_mm))
    same = bool(np.any(target_component & counterpart))
    authorized = np.zeros_like(truth, dtype=bool)
    if not same:
        goal = f"{operation}_NEW_COMPLETE"
        authorized = target_residual.copy()
    elif far_slice.any():
        goal = f"{operation}_SAME_COMPLETE"
        authorized = target_residual.copy()
    else:
        area_mm2 = float(local_slice.sum()) * spacing[0] * spacing[1]
        if area_mm2 < float(minimum_local_area_mm2):
            raise RuntimeError("SAME_LOCAL candidate is below minimum physical area")
        goal = f"{operation}_SAME_LOCAL"
        authorized[:, :, center_z] = local_slice
    if any(not authorized[tuple(int(v) for v in coord)] for coord in coordinates_xyz):
        raise RuntimeError("derived authorized target does not contain the scribble")
    return goal, authorized, {
        "target_component_voxels": int(target_component.sum()),
        "target_residual_voxels": int(target_residual.sum()),
        "operation": operation,
        "target": "SAME" if same else "NEW",
        "authorized_voxels": int(authorized.sum()),
        "far_slice_voxels": int(far_slice.sum()),
        "center_z": center_z,
        "scope_observation_plane": "prompted_axial_slice",
        "local_radius_mm": float(local_radius_mm),
    }


def write_binary_nifti(path: Path, mask: np.ndarray, reference: nib.Nifti1Image) -> None:
    header = reference.header.copy()
    header.set_data_dtype(np.uint8)
    nib.save(nib.Nifti1Image(mask.astype(np.uint8), reference.affine, header), str(path))


def mask_fits_physical_crop(
    mask_2d: np.ndarray,
    *,
    center_xy,
    spacing_xy,
    field_mm: float,
    output_size: int,
) -> bool:
    coordinates = np.argwhere(np.asarray(mask_2d) > 0)
    if not len(coordinates):
        return False
    output_spacing = float(field_mm) / int(output_size)
    limit = float(field_mm) / 2.0 - output_spacing / 2.0
    offsets = np.abs(
        (coordinates - np.asarray(center_xy, dtype=float)[None])
        * np.asarray(spacing_xy, dtype=float)[None]
    )
    return bool(np.all(offsets <= limit + 1e-6))


def _verified_source_path(
    row: Mapping[str, Any], path_key: str, hash_key: str
) -> Path:
    raw = Path(str(row.get(path_key) or ""))
    if raw.is_symlink():
        raise RuntimeError(f"{path_key} must be a regular non-symlink file")
    path = raw.resolve()
    if not path.is_file():
        raise RuntimeError(f"missing regular source artifact: {path_key}")
    if sha256_file(path) != row.get(hash_key):
        raise RuntimeError(f"source artifact hash mismatch: {path_key}")
    return path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--residual-manifest",
        type=Path,
        required=True,
        help="Output of build_petct_residual_manifest.py",
    )
    parser.add_argument(
        "--residual-ready",
        type=Path,
        help="Required hash-bound RESIDUAL_READY for the natural lane",
    )
    parser.add_argument("--official-simulator", type=Path, required=True)
    parser.add_argument(
        "--official-runtime-manifest",
        type=Path,
        default=DEFAULT_RUNTIME_MANIFEST,
    )
    parser.add_argument("--experiment-config", type=Path, required=True)
    parser.add_argument("--learning-split", type=Path, required=True)
    parser.add_argument(
        "--partitions",
        nargs="+",
        choices=("train", "val", "test"),
        required=True,
        help="explicit frozen learning partitions present in the residual manifest",
    )
    parser.add_argument("--official-commit")
    parser.add_argument("--strategy-mode", choices=["primary", "all"])
    parser.add_argument("--strategy-salt")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--lane", choices=["controlled", "natural"], required=True)
    parser.add_argument("--oof-ready", type=Path)
    parser.add_argument("--visible-root", type=Path, required=True)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--authorized-root", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--exclusions", type=Path, required=True)
    parser.add_argument("--ready-receipt", type=Path, required=True)
    add_leaf_test_access_arguments(parser)
    args = parser.parse_args(argv)
    if len(args.partitions) != len(set(args.partitions)):
        parser.error("--partitions must not contain duplicates")
    final_directories = [
        args.visible_root.resolve(),
        args.evaluation_root.resolve(),
        args.authorized_root.resolve(),
    ]
    final_files = [
        args.exclusions.resolve(),
        args.output_manifest.resolve(),
        args.ready_receipt.resolve(),
    ]
    selected_partitions = set(args.partitions)
    # The formal access decision must happen before reading the residual/OOF
    # manifests, simulator, source volumes, or output tree.
    try:
        test_access = enforce_partition_access(
            selected_partitions,
            receipt_path=args.test_access_receipt,
            experiment_config=args.experiment_config,
            learning_split=args.learning_split,
            run_root=args.run_root,
            output_paths=(*final_directories, *final_files),
        )
    except TestAccessError as error:
        parser.error(str(error))
    test_access_sha256 = (
        None if test_access is None else str(test_access["receipt_sha256"])
    )
    with args.experiment_config.open("r", encoding="utf-8") as stream:
        experiment_config = json.load(stream)
    if args.lane == "natural" and args.residual_ready is None:
        parser.error("natural lane requires --residual-ready")
    if args.lane == "natural" and args.oof_ready is None:
        parser.error("natural lane requires --oof-ready")
    residual_ready = None
    if args.lane == "natural":
        try:
            residual_ready = validate_residual_ready(
                args.residual_ready,
                residual_manifest=args.residual_manifest,
                oof_ready=args.oof_ready,
                selected_partitions=selected_partitions,
            )
        except RuntimeError as error:
            parser.error(str(error))
    residual_rows = load_jsonl(args.residual_manifest)
    try:
        validate_manifest_rows_against_frozen_learning_split(
            residual_rows,
            args.learning_split,
            require_episode_id=False,
            allowed_partitions=selected_partitions,
        )
    except LearningContractError as error:
        parser.error(str(error))
    for row_number, source in enumerate(residual_rows, start=1):
        expected_receipt = (
            test_access_sha256 if source.get("partition") == "test" else None
        )
        if source.get("test_access_receipt_sha256") != expected_receipt:
            parser.error(
                "residual row %d test-access receipt provenance mismatch"
                % row_number
            )
    if residual_ready is not None:
        buckets = residual_ready["validated_cohort"]
        manifest_case_ids = {str(row["case_id"]) for row in residual_rows}
        if len(manifest_case_ids) != len(residual_rows):
            parser.error("residual manifest contains duplicate case_id")
        if manifest_case_ids != set(buckets["generated"]["case_ids"]):
            parser.error("residual manifest differs from RESIDUAL_READY generated cohort")
        positive_from_rows = {
            str(row["case_id"])
            for row in residual_rows
            if int(row.get("fn_voxels", -1)) > 0
        }
        zero_from_rows = {
            str(row["case_id"])
            for row in residual_rows
            if int(row.get("fn_voxels", -1)) == 0
        }
        fp_positive_from_rows = {
            str(row["case_id"])
            for row in residual_rows
            if int(row.get("fp_voxels", -1)) > 0
        }
        zero_fp_from_rows = {
            str(row["case_id"])
            for row in residual_rows
            if int(row.get("fp_voxels", -1)) == 0
        }
        if positive_from_rows != set(buckets["fn_positive"]["case_ids"]):
            parser.error("residual manifest FN-positive cohort differs from RESIDUAL_READY")
        if zero_from_rows != set(buckets["zero_fn"]["case_ids"]):
            parser.error("residual manifest zero-FN cohort differs from RESIDUAL_READY")
        if fp_positive_from_rows != set(buckets["fp_positive"]["case_ids"]):
            parser.error("residual manifest FP-positive cohort differs from RESIDUAL_READY")
        if zero_fp_from_rows != set(buckets["zero_fp"]["case_ids"]):
            parser.error("residual manifest zero-FP cohort differs from RESIDUAL_READY")
        for row_number, source in enumerate(residual_rows, start=1):
            provenance = source.get("m0_provenance")
            if not isinstance(provenance, Mapping):
                parser.error(f"residual row {row_number} lacks M0 provenance")
            if provenance.get("input_gt_sha256") != source.get("gt_sha256"):
                parser.error(
                    f"residual row {row_number} GT hash differs from OOF provenance"
                )
    generation = resolve_scribble_generation_contract(
        experiment_config,
        official_commit=args.official_commit,
        strategy_mode=args.strategy_mode,
        strategy_salt=args.strategy_salt,
        seed=args.seed,
    )
    crop_config = experiment_config["learning_tensor_normalization"]
    if any(_path_lexists(path) for path in (*final_directories, *final_files)):
        parser.error("output paths must not already exist")
    simulator = load_official_simulator(
        args.official_simulator,
        expected_commit=generation["official_commit"],
        expected_sha256=generation["simulator_file_sha256"],
        runtime_manifest=args.official_runtime_manifest,
    )
    simulator_provenance = getattr(simulator, "_petct_official_provenance", None)
    validate_official_simulator_provenance(simulator_provenance, generation)
    validated_oof = None
    if args.lane == "natural":
        from baseline.validate_petct_m0_oof import validate_oof_ready_receipt_only

        validated_oof = validate_oof_ready_receipt_only(args.oof_ready)
    rows: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    patient_partition: Dict[str, str] = {}
    experiment_config_sha256 = sha256_file(args.experiment_config)
    local_radius_mm = float(experiment_config["editor"]["local_radius_mm"])
    minimum_local_area_mm2 = float(
        experiment_config["editor"]["minimum_local_area_mm2"]
    )
    visible_root, evaluation_root, authorized_root = final_directories
    exclusions_path, output_manifest, ready_receipt = final_files
    requested_attempts: dict[str, dict[str, str]] = {}
    for source in residual_rows:
        case_id = str(source["case_id"])
        patient_id = str(source["patient_id"]).casefold()
        partition = str(source["partition"])
        for operation in ("ADD", "REMOVE"):
            for requested_strategy in selected_strategies(
                patient_id,
                generation["strategy_mode"],
                generation["strategy_salt"],
            ):
                attempt_id = scribble_attempt_id(
                    args.lane, case_id, operation, requested_strategy
                )
                if attempt_id in requested_attempts:
                    raise RuntimeError("duplicate cue attempt id")
                requested_attempts[attempt_id] = {
                    "case_id": case_id,
                    "patient_id": patient_id,
                    "partition": partition,
                    "operation": operation,
                    "requested_strategy": requested_strategy,
                }

    def exclude_attempt(
        *,
        source: Mapping[str, Any],
        operation: str,
        requested_strategy: str,
        effective_strategy: str | None,
        reason: str,
        detail: str | None = None,
    ) -> None:
        item = {
            "case_id": str(source["case_id"]),
            "patient_id": str(source["patient_id"]).casefold(),
            "partition": str(source["partition"]),
            "attempt_id": scribble_attempt_id(
                args.lane, str(source["case_id"]), operation, requested_strategy
            ),
            "operation": operation,
            "requested_strategy": requested_strategy,
            "effective_strategy": effective_strategy,
            "reason": reason,
        }
        if detail:
            item["reason_detail"] = detail
        exclusions.append(item)

    with staged_output_bundle(
        directory_outputs=final_directories,
        file_outputs=final_files,
    ) as staged:
        staged_visible_root = staged[visible_root]
        staged_evaluation_root = staged[evaluation_root]
        staged_authorized_root = staged[authorized_root]
        for source in residual_rows:
            patient = str(source["patient_id"]).casefold()
            partition = str(source["partition"])
            if patient in patient_partition and patient_partition[patient] != partition:
                raise RuntimeError("patient crosses episode partitions")
            patient_partition[patient] = partition

            ct_path = _verified_source_path(source, "ct_path", "ct_sha256")
            pet_path = _verified_source_path(source, "pet_path", "pet_sha256")
            gt_path = _verified_source_path(source, "gt_path", "gt_sha256")
            m0_path = _verified_source_path(source, "m0_path", "m0_sha256")
            fn_path = _verified_source_path(source, "fn_path", "fn_sha256")
            fp_path = _verified_source_path(source, "fp_path", "fp_sha256")

            # The trust chain is deliberately ordered: M0 provenance first,
            # then exact FN, official simulator, lesion binding, and intent target.
            if args.lane == "natural":
                from baseline.validate_petct_m0_oof import build_natural_oof_binding_from_validated

                provenance = build_natural_oof_binding_from_validated(
                    validated_oof,
                    ready_path=args.oof_ready,
                    case_id=source["case_id"],
                    patient_id=patient,
                    m0_path=m0_path,
                    leaf_binding=source.get("truth_binding"),
                )
                if source.get("m0_provenance") != provenance:
                    raise RuntimeError("residual manifest OOF provenance changed")
            else:
                provenance = source.get("m0_provenance")
                if not isinstance(provenance, dict):
                    raise RuntimeError("controlled episode requires m0_provenance object")

            gt_image = nib.load(str(gt_path))
            m0_image = nib.load(str(m0_path))
            fn_image = nib.load(str(fn_path))
            fp_image = nib.load(str(fp_path))
            images = (m0_image, fn_image, fp_image)
            if any(
                image.shape != gt_image.shape
                or not np.allclose(image.affine, gt_image.affine, atol=1e-3, rtol=0)
                for image in images
            ):
                raise RuntimeError("GT/M0/FN/FP geometry mismatch")
            gt_array = np.asarray(gt_image.dataobj)
            m0_array = np.asarray(m0_image.dataobj)
            residual_assets = {
                "ADD": {
                    "kind": "FN",
                    "path": fn_path,
                    "sha256": str(source["fn_sha256"]),
                    "mask": verify_full_residual(
                        gt_array,
                        m0_array,
                        np.asarray(fn_image.dataobj),
                        operation="ADD",
                    ),
                },
                "REMOVE": {
                    "kind": "FP",
                    "path": fp_path,
                    "sha256": str(source["fp_sha256"]),
                    "mask": verify_full_residual(
                        gt_array,
                        m0_array,
                        np.asarray(fp_image.dataobj),
                        operation="REMOVE",
                    ),
                },
            }
            strategies = selected_strategies(
                patient,
                generation["strategy_mode"],
                generation["strategy_salt"],
            )
            for operation, asset in residual_assets.items():
                residual = asset["mask"]
                if not np.any(residual):
                    for strategy in strategies:
                        exclude_attempt(
                            source=source,
                            operation=operation,
                            requested_strategy=strategy,
                            effective_strategy=None,
                            reason=f"EMPTY_{asset['kind']}_RESIDUAL",
                        )
                    continue
                for strategy in strategies:
                    attempt_id = scribble_attempt_id(
                        args.lane, str(source["case_id"]), operation, strategy
                    )
                    try:
                        record = generate_residual_scribble(
                            residual,
                            operation=operation,
                            strategy=strategy,
                            simulator=simulator,
                            upstream_commit=generation["official_commit"],
                            seed=generation["seed"],
                            minimum_best_slice_pixels=generation[
                                "minimum_best_slice_pixels"
                            ],
                        )
                    except ResidualCueIneligibleError as exc:
                        # A residual too small to draw on is a reasoned, counted
                        # exclusion in every strategy mode.  It is decided before
                        # the simulator runs, so it never masks a contract breach:
                        # a cue that lands outside an ELIGIBLE component still
                        # raises below and still fails the run closed.
                        exclude_attempt(
                            source=source,
                            operation=operation,
                            requested_strategy=strategy,
                            effective_strategy=None,
                            reason=CUE_INELIGIBLE_REASON,
                            detail=str(exc),
                        )
                        continue
                    except Exception as exc:
                        if generation["strategy_mode"] == "primary":
                            raise RuntimeError(
                                f"primary cue attempt {attempt_id} failed closed: {exc}"
                            ) from exc
                        exclude_attempt(
                            source=source,
                            operation=operation,
                            requested_strategy=strategy,
                            effective_strategy=None,
                            reason="CUE_GENERATION_FAILED",
                            detail=str(exc),
                        )
                        continue
                    disposition = apply_strategy_identity_policy(
                        record,
                        strategy_mode=generation["strategy_mode"],
                        context=f"primary cue attempt {attempt_id}",
                    )
                    if disposition is not None:
                        exclude_attempt(
                            source=source,
                            operation=operation,
                            requested_strategy=strategy,
                            effective_strategy=disposition["effective_strategy"],
                            reason=disposition["reason"],
                            detail=disposition["detail"],
                        )
                        continue
                    generation_receipt = {
                        **generation,
                        "attempt_id": attempt_id,
                        "operation": operation,
                        "residual_kind": asset["kind"],
                        "selected_strategy": strategy,
                        "requested_strategy": record["requested_strategy"],
                        "effective_strategy": record["effective_strategy"],
                        "strategy_fallback": record["strategy_fallback"],
                        "fallback_reason": record["fallback_reason"],
                        "strategy_audit": record["strategy_audit"],
                        "official_source_provenance": dict(simulator_provenance),
                        "m0_provenance": provenance,
                        "cue_contract_version": record["contract_version"],
                        "simulator_entrypoint": record["simulator_entrypoint"],
                        "residual_artifact_path": str(asset["path"]),
                        "residual_artifact_sha256": asset["sha256"],
                        "residual_mask_sha256": record["residual_sha256"],
                        "residual_voxels": record["residual_voxels"],
                        "cue_eligibility": record["cue_eligibility"],
                        "eligible_residual_sha256": record["eligible_residual_sha256"],
                        "coordinate_count": record["coordinate_count"],
                        "coordinate_sha256": record["coordinate_sha256"],
                        "source_slice": record["source_slice"],
                        "source_component_area": record["source_component_area"],
                        "cue_polarity": record["polarity"],
                        "single_component_connectivity": 18,
                        "local_radius_mm": local_radius_mm,
                        "minimum_local_area_mm2": minimum_local_area_mm2,
                        "crop_field_mm": float(crop_config["crop_field_mm"]),
                        "crop_output_size_px": int(crop_config["output_size_px"]),
                        "crop_output_spacing_mm": float(crop_config["output_spacing_mm"]),
                        "experiment_config_sha256": experiment_config_sha256,
                    }
                    record["official_source_provenance"] = dict(simulator_provenance)
                    record["generation_receipt"] = generation_receipt
                    try:
                        goal, authorized, target_stats = derive_goal_and_authorized_target(
                            gt=gt_array,
                            m0=m0_array,
                            operation=operation,
                            coordinates_xyz=record["coordinates_xyz"],
                            spacing_xy=gt_image.header.get_zooms()[:2],
                            local_radius_mm=local_radius_mm,
                            minimum_local_area_mm2=minimum_local_area_mm2,
                        )
                    except RuntimeError as exc:
                        exclude_attempt(
                            source=source,
                            operation=operation,
                            requested_strategy=strategy,
                            effective_strategy=str(record["effective_strategy"]),
                            reason=classify_derivation_refusal(exc),
                            detail=str(exc),
                        )
                        continue
                    center_xy = np.mean(
                        np.asarray([[c[0], c[1]] for c in record["coordinates_xyz"]]),
                        axis=0,
                    )
                    center_z = int(record["coordinates_xyz"][0][2])
                    if not mask_fits_physical_crop(
                        authorized[:, :, center_z],
                        center_xy=center_xy,
                        spacing_xy=gt_image.header.get_zooms()[:2],
                        field_mm=float(crop_config["crop_field_mm"]),
                        output_size=int(crop_config["output_size_px"]),
                    ):
                        exclude_attempt(
                            source=source,
                            operation=operation,
                            requested_strategy=strategy,
                            effective_strategy=str(record["effective_strategy"]),
                            reason="AUTHORIZED_TARGET_EXCEEDS_FROZEN_PHYSICAL_CROP",
                        )
                        continue
                    episode_id = opaque_episode_id(str(source["case_id"]), goal, strategy)
                    authorized_staged_path = staged_authorized_root / f"{episode_id}_authorized.nii.gz"
                    authorized_final_path = authorized_root / authorized_staged_path.name
                    write_binary_nifti(authorized_staged_path, authorized, gt_image)
                    authorized_sha256 = sha256_file(authorized_staged_path)
                    generation_receipt.update(
                        {
                            "goal": goal,
                            "target_stats": target_stats,
                            "authorized_target_sha256": authorized_sha256,
                        }
                    )
                    patient_hash = hashlib.sha256(
                        ("PETCT-PATIENT-GROUP-v2|" + patient).encode("utf-8")
                    ).hexdigest()
                    visible, evaluation = build_episode_documents(
                        episode_id=episode_id,
                        lane=args.lane,
                        patient_group_hash=patient_hash,
                        montage_reference=f"learning-visible/{episode_id}.npz",
                        m0_provenance=provenance,
                        scribble_record=record,
                        source_case_id=str(source["case_id"]),
                        source_patient_id=patient,
                        residual_sha256=record["residual_sha256"],
                        residual_voxels=record["residual_voxels"],
                        gold_intent=canonical_intent_frame(goal),
                    )
                    receipt = publish_episode_documents(
                        visible,
                        evaluation,
                        visible_root=staged_visible_root,
                        eval_root=staged_evaluation_root,
                    )
                    visible_final_path = visible_root / f"{episode_id}.json"
                    evaluation_final_path = evaluation_root / f"{episode_id}.json"
                    rows.append(
                        {
                            **{
                                key: source[key]
                                for key in ("case_id", "patient_id", "partition", "held_out_fold")
                            },
                            "ct_path": str(ct_path),
                            "pet_path": str(pet_path),
                            "m0_path": str(m0_path),
                            "gt_path": str(gt_path),
                            **{
                                key: source[key]
                                for key in (
                                    "ct_sha256", "pet_sha256", "m0_sha256",
                                    "gt_sha256", "learning_split_sha256",
                                )
                            },
                            "authorized_path": str(authorized_final_path),
                            "authorized_sha256": authorized_sha256,
                            "episode_id": episode_id,
                            "attempt_id": attempt_id,
                            "goal": goal,
                            "operation": operation,
                            "target": target_stats["target"],
                            "scope": goal.rsplit("_", 1)[1],
                            "cue_polarity": record["polarity"],
                            "strategy": strategy,
                            "requested_strategy": record["requested_strategy"],
                            "effective_strategy": record["effective_strategy"],
                            "strategy_fallback": record["strategy_fallback"],
                            "strategy_mode": generation["strategy_mode"],
                            "strategy_salt": generation["strategy_salt"],
                            "strategy_assignment": generation["strategy_assignment"],
                            "seed": generation["seed"],
                            "coordinates_xyz": record["coordinates_xyz"],
                            "visible_document": str(visible_final_path),
                            "visible_document_sha256": receipt["visible_sha256"],
                            "evaluation_document": str(evaluation_final_path),
                            "evaluation_document_sha256": receipt["eval_sha256"],
                            "scribble_density_mode": record["scribble_density_mode"],
                            "fallback_mode": record["fallback_mode"],
                            "m0_provenance": provenance,
                            "residual_kind": asset["kind"],
                            "residual_path": str(asset["path"]),
                            "residual_sha256": asset["sha256"],
                            "residual_voxels": int(record["residual_voxels"]),
                            "residual_mask_sha256": str(record["residual_sha256"]),
                            "fn_path": str(fn_path),
                            "fn_sha256": str(source["fn_sha256"]),
                            "fp_path": str(fp_path),
                            "fp_sha256": str(source["fp_sha256"]),
                            "residual_contract": source.get("residual_contract"),
                            "official_source_provenance": dict(simulator_provenance),
                            "scribble_generation": generation_receipt,
                            "experiment_config_sha256": experiment_config_sha256,
                            "test_access_receipt_sha256": (
                                test_access_sha256 if partition == "test" else None
                            ),
                            "target_stats": target_stats,
                        }
                    )
        with staged[exclusions_path].open(
            "x", encoding="utf-8", newline="\n"
        ) as stream:
            for row in exclusions:
                stream.write(json.dumps(row, sort_keys=True) + "\n")
        with staged[output_manifest].open(
            "x", encoding="utf-8", newline="\n"
        ) as stream:
            for row in rows:
                stream.write(json.dumps(row, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

        generated_attempt_ids = {str(row["attempt_id"]) for row in rows}
        excluded_attempt_ids = {str(row["attempt_id"]) for row in exclusions}
        if len(excluded_attempt_ids) != len(exclusions):
            raise RuntimeError("one scribble attempt received multiple exclusions")
        if generated_attempt_ids & excluded_attempt_ids:
            raise RuntimeError("scribble attempt is both generated and excluded")
        if generated_attempt_ids | excluded_attempt_ids != set(requested_attempts):
            raise RuntimeError("scribble attempt denominator is not closed")
        if not generated_attempt_ids:
            raise RuntimeError("no eligible scribble attempts were produced")
        generated_cases = {str(row["case_id"]) for row in rows}
        generated_patients = {str(row["patient_id"]).casefold() for row in rows}
        cases_with_excluded_attempts = {
            str(row["case_id"]) for row in exclusions
        }
        patients_with_excluded_attempts = {
            str(row["patient_id"]).casefold() for row in exclusions
        }
        selected_cases = {str(row["case_id"]) for row in residual_rows}
        selected_patients = {
            str(row["patient_id"]).casefold() for row in residual_rows
        }
        fully_excluded_cases = selected_cases - generated_cases
        fully_excluded_patients = {
            str(row["patient_id"]).casefold()
            for row in residual_rows
            if str(row["case_id"]) in fully_excluded_cases
        }
        partially_excluded_cases = cases_with_excluded_attempts & generated_cases
        partially_excluded_patients = {
            str(row["patient_id"]).casefold()
            for row in exclusions
            if str(row["case_id"]) in partially_excluded_cases
        }
        if residual_ready is None:
            source_bucket = _cohort_bucket(selected_cases, selected_patients)
        else:
            source_bucket = residual_ready["validated_cohort"]["source"]
        ready = {
            "schema_version": SCRIBBLE_READY_SCHEMA,
            "status": "PASS",
            "phase": SCRIBBLE_READY_PHASE,
            "lane": args.lane,
            "strategy_mode": generation["strategy_mode"],
            "selected_partitions": sorted(selected_partitions),
            "inputs": {
                "residual_manifest": _output_file_record(
                    args.residual_manifest.resolve(), args.residual_manifest.resolve()
                ),
                "residual_ready": (
                    None
                    if residual_ready is None
                    else {
                        "path": residual_ready["ready_path"],
                        "bytes": Path(residual_ready["ready_path"]).stat().st_size,
                        "sha256": residual_ready["ready_sha256"],
                    }
                ),
                "oof_ready": (
                    None
                    if args.oof_ready is None
                    else _output_file_record(
                        args.oof_ready.resolve(), args.oof_ready.resolve()
                    )
                ),
                "experiment_config": _output_file_record(
                    args.experiment_config.resolve(), args.experiment_config.resolve()
                ),
                "learning_split": _output_file_record(
                    args.learning_split.resolve(), args.learning_split.resolve()
                ),
                "official_source_provenance": dict(simulator_provenance),
            },
            "outputs": {
                "manifest": _output_file_record(
                    staged[output_manifest], output_manifest
                ),
                "exclusions": _output_file_record(
                    staged[exclusions_path], exclusions_path
                ),
                "visible": _tree_record(staged_visible_root, visible_root),
                "evaluation": _tree_record(
                    staged_evaluation_root, evaluation_root
                ),
                "authorized": _tree_record(
                    staged_authorized_root, authorized_root
                ),
            },
            "cohort": {
                "source": source_bucket,
                "selected_source": _cohort_bucket(
                    selected_cases, selected_patients
                ),
                "eligible": _cohort_bucket(
                    generated_cases, generated_patients
                ),
                "excluded": _cohort_bucket(
                    fully_excluded_cases, fully_excluded_patients
                ),
                "partially_excluded": _cohort_bucket(
                    partially_excluded_cases, partially_excluded_patients
                ),
                "with_excluded_attempts": _cohort_bucket(
                    cases_with_excluded_attempts,
                    patients_with_excluded_attempts,
                ),
            },
            "attempts": {
                "requested_count": len(requested_attempts),
                "requested_ids": sorted(requested_attempts),
                "requested_ids_sha256": _canonical_hash(sorted(requested_attempts)),
                "generated_count": len(generated_attempt_ids),
                "generated_ids": sorted(generated_attempt_ids),
                "generated_ids_sha256": _canonical_hash(
                    sorted(generated_attempt_ids)
                ),
                "excluded_count": len(excluded_attempt_ids),
                "excluded_ids": sorted(excluded_attempt_ids),
                "excluded_ids_sha256": _canonical_hash(
                    sorted(excluded_attempt_ids)
                ),
                "episodes": len(rows),
            },
            "exclusions_by_reason": _exclusion_summary(exclusions),
            "survivor_coverage": {
                "attempt_fraction": (
                    len(generated_attempt_ids) / len(requested_attempts)
                    if requested_attempts
                    else 0.0
                ),
                "case_fraction": (
                    len(generated_cases) / len(selected_cases)
                    if selected_cases
                    else 0.0
                ),
                "patient_fraction": (
                    len(generated_patients) / len(selected_patients)
                    if selected_patients
                    else 0.0
                ),
            },
            "experiment_result_count": 0,
            "thesis_citable": False,
        }
        ready["binding_sha256"] = _canonical_hash(ready)
        with staged[ready_receipt].open(
            "x", encoding="utf-8", newline="\n"
        ) as stream:
            stream.write(json.dumps(ready, ensure_ascii=False, indent=2, sort_keys=True))
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
    print(
        json.dumps(
            {
                "episodes": len(rows),
                "excluded_cases": len(exclusions),
                "strategies": generation["strategy_mode"],
                "ready_receipt": str(ready_receipt),
                "ready_receipt_sha256": sha256_file(ready_receipt),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
