from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "scripts"))

from data.build_petct_scribble_dataset import (  # noqa: E402
    _exclusion_summary,
    apply_strategy_identity_policy,
    resolve_scribble_generation_contract,
    scribble_attempt_id,
    staged_output_bundle,
    validate_official_simulator_provenance,
    validate_residual_ready,
    verify_full_residual,
)
from data.build_petct_scribble_episode import (  # noqa: E402
    AUTOPETV_RUNTIME_SCHEMA,
    AUTOPETV_RUNTIME_STATUS,
    CUE_ELIGIBILITY_RULE,
    CUE_INELIGIBLE_REASON,
    DEFAULT_RUNTIME_MANIFEST_SHA256,
    official_simulator_provenance,
)


def _autopetv_runtime_root_or_skip() -> Path:
    candidates = (
        PROJECT / "external_runners" / "autopetv_protocol",
        PROJECT / "upstream" / "autoPETV",
    )
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    pytest.skip(
        "requires the pinned autoPETV minimal runtime or upstream checkout; "
        "neither vendor asset directory is present"
    )


def test_v2_generation_contract_is_bidirectional_and_exact() -> None:
    config = json.loads(
        (PROJECT / "configs" / "petct_route_a_experiment.json").read_text(
            encoding="utf-8"
        )
    )
    primary = resolve_scribble_generation_contract(config)
    robustness = resolve_scribble_generation_contract(config, strategy_mode="all")
    assert primary["polarity_contract"]["ADD"]["source_residual"] == "FN"
    assert primary["polarity_contract"]["REMOVE"]["source_residual"] == "FP"
    assert robustness["strategy_mode"] == "all"
    assert "three-matched-attempts" in robustness["robustness_strategy_contract"]


def test_current_v2_runtime_manifest_provenance_flows_into_dataset_and_v1_fails_closed(
    tmp_path: Path,
) -> None:
    runtime_root = tmp_path / "autopetv-minimal-runtime"
    (runtime_root / "interactive").mkdir(parents=True)
    upstream = _autopetv_runtime_root_or_skip()
    for relative in (
        "LICENSE",
        "interactive/simulate_scribbles.py",
        "metrics.py",
    ):
        source = upstream / relative
        target = runtime_root / relative
        target.write_bytes(source.read_bytes())

    config = json.loads(
        (PROJECT / "configs" / "petct_route_a_experiment.json").read_text(
            encoding="utf-8"
        )
    )
    generation = resolve_scribble_generation_contract(config)
    provenance = official_simulator_provenance(
        runtime_root / "interactive" / "simulate_scribbles.py",
        expected_commit=generation["official_commit"],
        expected_sha256=generation["simulator_file_sha256"],
        runtime_manifest=PROJECT / "protocols" / "autopetv_protocol_runtime.json",
    )
    assert provenance["runtime_manifest"] == {
        "path": str(
            (PROJECT / "protocols" / "autopetv_protocol_runtime.json").resolve()
        ),
        "sha256": DEFAULT_RUNTIME_MANIFEST_SHA256,
        "schema_version": AUTOPETV_RUNTIME_SCHEMA,
        "status": AUTOPETV_RUNTIME_STATUS,
    }
    validate_official_simulator_provenance(provenance, generation)

    legacy = copy.deepcopy(provenance)
    legacy["runtime_manifest"].update(
        {
            "schema_version": "PETCT-AUTOPETV-PROTOCOL-RUNTIME-v1.0",
            "status": "FROZEN_MINIMAL_RUNTIME",
            "sha256": (
                "c1674081dbac0382e58a15eccf4da0b99a05b526860219fb53b762c8d3662d5a"
            ),
        }
    )
    with pytest.raises(RuntimeError, match="manifest provenance mismatch"):
        validate_official_simulator_provenance(legacy, generation)


def test_full_residual_verifier_accepts_exact_fn_and_fp_only() -> None:
    gt = np.zeros((6, 6, 2), dtype=np.uint8)
    m0 = np.zeros_like(gt)
    gt[1:4, 1:4, 0] = 1
    m0[2:5, 2:5, 0] = 1
    fn = (gt > 0) & ~(m0 > 0)
    fp = (m0 > 0) & ~(gt > 0)
    assert np.array_equal(verify_full_residual(gt, m0, fn, operation="ADD"), fn)
    assert np.array_equal(verify_full_residual(gt, m0, fp, operation="REMOVE"), fp)
    with pytest.raises(RuntimeError, match="exactly"):
        verify_full_residual(gt, m0, fp, operation="ADD")


def test_attempt_ids_separate_operation_and_reject_unknown_polarity() -> None:
    assert scribble_attempt_id("natural", "case", "ADD", "random") != (
        scribble_attempt_id("natural", "case", "REMOVE", "random")
    )
    with pytest.raises(ValueError, match="operation"):
        scribble_attempt_id("natural", "case", "UPDATE", "random")


def test_requested_effective_strategy_crossing_is_never_silently_relabelled() -> None:
    record = {
        "requested_strategy": "centerline",
        "effective_strategy": "random",
        "strategy_fallback": True,
        "fallback_reason": "UPSTREAM_EXCEPTION_FALLBACK_TO_RANDOM",
    }
    with pytest.raises(RuntimeError, match="failed closed"):
        apply_strategy_identity_policy(record, strategy_mode="primary", context="test")
    disposition = apply_strategy_identity_policy(
        record, strategy_mode="all", context="test"
    )
    assert disposition == {
        "effective_strategy": "random",
        "reason": "CROSS_STRATEGY_FALLBACK",
        "detail": "UPSTREAM_EXCEPTION_FALLBACK_TO_RANDOM",
    }


def test_staging_bundle_rejects_overlap_existing_and_rolls_back(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError):
        with staged_output_bundle(
            directory_outputs=[tmp_path / "output"],
            file_outputs=[tmp_path / "output" / "manifest.json"],
        ):
            pass
    existing = tmp_path / "ready.json"
    existing.write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError, match="existing"):
        with staged_output_bundle(directory_outputs=[], file_outputs=[existing]):
            pass
    final_dir = tmp_path / "new-dir"
    final_file = tmp_path / "new.json"
    with pytest.raises(ValueError, match="abort"):
        with staged_output_bundle(
            directory_outputs=[final_dir], file_outputs=[final_file]
        ) as staged:
            staged[final_file].write_text("partial", encoding="utf-8")
            raise ValueError("abort")
    assert not final_dir.exists() and not final_file.exists()


def test_residual_ready_rejects_symlink_before_json_or_hash_access(tmp_path: Path) -> None:
    target = tmp_path / "target.json"
    target.write_text("not-json", encoding="utf-8")
    link = tmp_path / "ready.json"
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("symlink creation unavailable")
    with pytest.raises(RuntimeError, match="non-symlink"):
        validate_residual_ready(
            link,
            residual_manifest=tmp_path / "residual.jsonl",
            oof_ready=tmp_path / "oof.json",
            selected_partitions={"train"},
        )


def _experiment_config() -> dict:
    return json.loads(
        (PROJECT / "configs" / "petct_route_a_experiment.json").read_text(
            encoding="utf-8"
        )
    )


def test_frozen_config_pins_the_pixel_unit_cue_eligibility_threshold() -> None:
    generation = resolve_scribble_generation_contract(_experiment_config())
    assert generation["minimum_best_slice_pixels"] == 5
    assert generation["cue_eligibility_rule"] == CUE_ELIGIBILITY_RULE


@pytest.mark.parametrize(
    ("key", "value"),
    (
        ("minimum_best_slice_pixels", 0),
        ("minimum_best_slice_pixels", True),
        ("minimum_best_slice_pixels", 5.0),
        ("minimum_best_slice_pixels", None),
        ("cue_eligibility_rule", "minimum-local-area-mm2"),
        ("cue_eligibility_rule", None),
    ),
)
def test_cue_eligibility_config_drift_fails_closed(key: str, value) -> None:
    config = _experiment_config()
    if value is None:
        config["scribble"].pop(key)
    else:
        config["scribble"][key] = value
    with pytest.raises(RuntimeError, match="minimum_best_slice_pixels|eligibility"):
        resolve_scribble_generation_contract(config)


def test_ineligible_cue_exclusions_are_counted_under_a_named_reason() -> None:
    exclusions = [
        {
            "attempt_id": "scribble-attempt-aaa",
            "reason": CUE_INELIGIBLE_REASON,
        },
        {
            "attempt_id": "scribble-attempt-bbb",
            "reason": CUE_INELIGIBLE_REASON,
        },
        {"attempt_id": "scribble-attempt-ccc", "reason": "EMPTY_FN_RESIDUAL"},
    ]
    summary = _exclusion_summary(exclusions)
    assert summary[CUE_INELIGIBLE_REASON]["count"] == 2
    assert summary[CUE_INELIGIBLE_REASON]["attempt_ids"] == [
        "scribble-attempt-aaa",
        "scribble-attempt-bbb",
    ]
    assert summary["EMPTY_FN_RESIDUAL"]["count"] == 1


def test_ineligible_cue_is_excluded_while_other_failures_stay_fail_closed() -> None:
    source = (
        PROJECT / "scripts" / "data" / "build_petct_scribble_dataset.py"
    ).read_text(encoding="utf-8")
    ineligible = source.index("except ResidualCueIneligibleError as exc:")
    generic = source.index("except Exception as exc:", ineligible)
    failed_closed = source.index("primary cue attempt", generic)
    # The narrow eligibility handler must be reached first, and the generic
    # handler must still fail the run closed in primary strategy mode.
    assert ineligible < generic < failed_closed


def test_partition_access_gate_precedes_manifest_and_volume_reads() -> None:
    source = (
        PROJECT / "scripts" / "data" / "build_petct_scribble_dataset.py"
    ).read_text(encoding="utf-8")
    gate = source.index("enforce_partition_access(")
    residual_gate = source.index("residual_ready = validate_residual_ready")
    manifest_read = source.index("residual_rows = load_jsonl")
    assert gate < residual_gate < manifest_read


# --- refusal classification -------------------------------------------------
#
# A single broad `except RuntimeError` used to record all eight derivation
# failures as one exclusion reason, so "this episode is not eligible" and "the
# derivation contradicted itself" were indistinguishable in the receipts.


def test_legitimate_refusals_each_keep_their_own_reason() -> None:
    from data.build_petct_scribble_dataset import (  # noqa: PLC0415
        DERIVATION_REFUSAL_REASONS,
        classify_derivation_refusal,
    )

    seen = set()
    for fragment, reason in DERIVATION_REFUSAL_REASONS.items():
        assert classify_derivation_refusal(RuntimeError(fragment)) == reason
        seen.add(reason)
    assert len(seen) == len(DERIVATION_REFUSAL_REASONS), "reasons must be distinct"
    assert "STATE_RELATIVE_TARGET_INELIGIBLE" not in seen


def test_internal_contract_violations_stop_the_build_instead_of_excluding() -> None:
    from data.build_petct_scribble_dataset import (  # noqa: PLC0415
        DERIVATION_HARD_FAILURES,
        classify_derivation_refusal,
    )

    for fragment in DERIVATION_HARD_FAILURES:
        with pytest.raises(RuntimeError, match="refusing to record"):
            classify_derivation_refusal(RuntimeError(fragment))

    # The most serious one is specifically covered: an authorized target that
    # does not contain its own scribble means the derivation is broken.
    assert (
        "derived authorized target does not contain the scribble"
        in DERIVATION_HARD_FAILURES
    )


def test_unrecognised_refusal_is_never_swallowed() -> None:
    from data.build_petct_scribble_dataset import (  # noqa: PLC0415
        classify_derivation_refusal,
    )

    with pytest.raises(RuntimeError, match="unclassified"):
        classify_derivation_refusal(RuntimeError("something nobody has seen before"))


def test_every_raise_site_in_the_derivation_is_classified() -> None:
    """New failure modes must be triaged, not silently fall through."""

    import re  # noqa: PLC0415

    from data.build_petct_scribble_dataset import (  # noqa: PLC0415
        DERIVATION_HARD_FAILURES,
        DERIVATION_REFUSAL_REASONS,
    )

    source = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "data"
        / "build_petct_scribble_dataset.py"
    ).read_text(encoding="utf-8")
    body = source.split("def derive_goal_and_authorized_target(", 1)[1]
    body = body.split("\ndef ", 1)[0]
    messages = re.findall(r'raise RuntimeError\(\s*"([^"]+)"', body)
    assert messages, "could not read the derivation's raise sites"

    known = set(DERIVATION_HARD_FAILURES) | set(DERIVATION_REFUSAL_REASONS)
    for message in messages:
        assert message in known, f"unclassified derivation failure: {message}"
