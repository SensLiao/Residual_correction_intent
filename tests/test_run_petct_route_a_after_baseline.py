from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT / "scripts" / "orchestration" / "run_petct_route_a_after_baseline.sh"
VALIDATOR = (
    PROJECT
    / "scripts"
    / "orchestration"
    / "validate_petct_route_a_receipt_pipeline.py"
)


def source() -> str:
    return SCRIPT.read_text(encoding="utf-8")


def test_launcher_is_fail_closed_and_consumes_external_final_freeze_once() -> None:
    shell = source()
    assert "set -euo pipefail" in shell
    assert "--partition test" in shell
    assert "--allow-test-access" not in shell
    assert "--final-freeze-grant" in shell
    assert 'EVALUATION_PARTITION}" == "test" && -z "${FINAL_FREEZE_GRANT}"' in shell
    assert 'EVALUATION_PARTITION}" == "val" && -n "${FINAL_FREEZE_GRANT}"' in shell
    assert shell.count('petct_test_access.py" consume') == 1
    assert 'petct_test_access.py" grant' not in shell
    assert "--test-access-receipt" in shell
    assert "exit 20" in shell
    assert "PETCT-ROUTE-A-BLOCKED-v1.0" not in shell
    assert "exit 42" not in shell
    consume = shell.index('petct_test_access.py" consume')
    assert consume < shell.index("run_stage environment_preflight")
    assert consume < shell.index("run_stage validate_oof_ready")
    assert consume < shell.index("run_stage source_identity_manifest")
    assert consume < shell.index("run_stage source_manifest")


def test_launcher_preserves_scientific_stage_order() -> None:
    shell = source()
    ordered = [
        "run_stage validate_oof_ready",
        "run_stage source_identity_manifest",
        "run_stage learning_split",
        "run_stage validate_learning_split",
        "run_stage source_manifest",
        "run_stage m0_metrics",
        "run_stage validate_m0_evaluation",
        "run_stage residual_manifest",
        "run_stage controlled_matched_states",
        "run_stage natural_autopetv_scribbles",
        "run_stage controlled_tensors",
        "run_stage natural_tensors",
        "run_stage validate_p2t_data",
        "run_stage validate_editor_data",
        'run_p2t_train_queue "${GPU0_ID}" 0 &',
        'run_p2t_eval_queue "${GPU0_ID}" 0 &',
        "run_stage p2t_confirmatory",
        "run_stage validate_p2t_results",
        'run_p2t_natural_queue "${GPU0_ID}" 0 &',
        'run_editor_train_queue "${GPU0_ID}" 0 &',
        "run_stage intent_interventions",
        'run_editor_eval_queue "${GPU0_ID}" 0 &',
        "run_stage editor_confirmatory",
        "run_stage validate_editor_results",
        "run_stage validate_complete",
        "run_stage robustness_all_corpus",
    ]
    offsets = [shell.index(token) for token in ordered]
    assert offsets == sorted(offsets)


def test_launcher_uses_metadata_only_oof_pre_gate() -> None:
    shell = source()
    assert 'validate-ready-receipt "${OOF_READY}"' in shell
    assert 'validate-ready "${OOF_READY}"' not in shell


def test_launcher_freezes_identity_before_partition_scoped_leaf_materialization() -> None:
    shell = source()
    identity_start = shell.index("run_stage source_identity_manifest")
    identity = shell[
        identity_start : shell.index(
            'if [[ "${EVALUATION_PARTITION}" == "val" ]]', identity_start
        )
    ]
    materialized = shell[shell.index("run_stage source_manifest") : shell.index("M0_ROWS=")]
    assert "--mode identity" in identity
    assert '--output "${IDENTITY_CASE_MANIFEST}"' in identity
    assert '--case-manifest "${IDENTITY_CASE_MANIFEST}"' in shell
    assert "--mode materialize" in materialized
    assert '--identity-manifest "${IDENTITY_CASE_MANIFEST}"' in materialized
    assert '--partitions "${SELECTED_PARTITIONS[@]}"' in materialized
    assert '"${TEST_ACCESS_ARGS[@]}"' in materialized
    assert '--output "${CASE_MANIFEST}"' in materialized


def test_launcher_wires_only_real_clis_and_two_serial_gpu_queues() -> None:
    shell = source()
    for relative in (
        "baseline/run_petct_m0_oof_parallel.sh",
        "data/build_petct_source_case_manifest.py",
        "evaluation/evaluate_petct_m0_oof.py",
        "data/build_petct_learning_split.py",
        "data/build_petct_residual_manifest.py",
        "p2t/build_petct_matched_state_dataset.py",
        "data/build_petct_scribble_dataset.py",
        "data/materialize_petct_learning_tensors.py",
        "p2t/train_petct_p2t.py",
        "p2t/infer_petct_p2t.py",
        "evaluation/evaluate_petct_p2t.py",
        "editor/train_petct_residual_editor.py",
        "editor/build_petct_intent_interventions.py",
        "editor/infer_petct_residual_editor.py",
        "evaluation/evaluate_petct_correction.py",
        "evaluation/aggregate_petct_condition_metrics.py",
        "evaluation/aggregate_petct_p2t_confirmatory.py",
        "orchestration/validate_petct_route_a_receipt_pipeline.py",
        "data/freeze_petct_intent_taxonomy.py",
    ):
        assert relative in shell
    assert 'run_p2t_train_queue "${GPU0_ID}" 0 &' in shell
    assert 'run_p2t_train_queue "${GPU1_ID}" 1 &' in shell
    assert 'run_p2t_eval_queue "${GPU0_ID}" 0 &' in shell
    assert 'run_p2t_eval_queue "${GPU1_ID}" 1 &' in shell
    assert 'run_editor_train_queue "${GPU0_ID}" 0 &' in shell
    assert 'run_editor_train_queue "${GPU1_ID}" 1 &' in shell
    assert 'run_editor_eval_queue "${GPU0_ID}" 0 &' in shell
    assert 'run_editor_eval_queue "${GPU1_ID}" 1 &' in shell
    assert 'wait "${P2T_GPU0_PID}"' in shell
    assert 'wait "${P2T_NATURAL_GPU1_PID}"' in shell
    assert 'wait "${EDITOR_GPU1_PID}"' in shell
    assert 'EDITOR_PRIMARY_ARCHITECTURE="$(' in shell
    assert "run_stage p2t_descriptive" in shell
    assert "run_stage editor_descriptive" in shell


def test_launcher_stops_before_deferred_mpsl_lane() -> None:
    shell = source()
    assert 'P2T_SECONDARY_ARCHITECTURE=' not in shell
    assert "mpsl_inspired_global_pool_v1" not in shell
    assert "validate_petct_p2t_secondary_results.py" not in shell
    assert "P2T_SECONDARY_RESULTS_READY.json" not in shell
    assert "P2T_SECONDARY_RESULTS_FAILED.json" not in shell
    assert "models/p2t_secondary" not in shell
    assert shell.index('run_p2t_natural_queue "${GPU0_ID}" 0 &') < shell.index(
        "run_stage validate_complete"
    )
    stop = shell.index("Route A six-class simple-first campaign complete")
    assert "exit 0" in shell[stop:]
    assert shell.rstrip().endswith("exit 0")


def test_launcher_trains_and_evaluates_visual_state_only_as_independent_checkpoint() -> None:
    shell = source()
    assert '["editor"]["training_conditions"]' in shell
    assert "TRAINABLE_EDITOR_CONDITIONS=(visual_state_only" not in shell
    assert 'EXECUTABLE_EDITOR_CONDITIONS=("${EDITOR_CONDITIONS[@]}")' in shell
    assert '--output "${RUN_ROOT}/models/editor/${condition}_seed${seed}.pth"' in shell
    assert 'trained="$(editor_checkpoint_condition "${condition}")"' in shell
    assert 'checkpoint="$(editor_checkpoint_path "${trained}" "${seed}")"' in shell
    assert '*) printf \'%s\\n\' "$1" ;;' in shell
    assert "--fusion-mode" not in shell[: shell.index("exit 0", shell.index("Route A six-class simple-first campaign complete"))]


def test_launcher_validates_receipts_and_aggregates_only_after_all_conditions() -> None:
    shell = source()
    assert "--target p2t_data" in shell
    assert "--target editor_data" in shell
    assert "--target p2t_results" in shell
    assert shell.index("--target p2t_data") < shell.index("run_p2t_train_queue")
    assert shell.index("--target editor_data") < shell.index("run_editor_train_queue")
    assert shell.index("--target p2t_results") < shell.index("run_editor_train_queue")
    assert "--target editor_results" in shell
    assert "--target complete" in shell
    assert 'aggregate_petct_condition_metrics.py" \\' in shell
    assert "predicted_slots" in shell
    assert "oracle_slots|predicted_slots) printf '%s\\n' scribble_plus_intent" in shell
    assert '--training-manifest "${training_manifest}"' in shell
    assert '--manifest "${NATURAL_TENSORS}"' in shell
    assert shell.count("run_stage editor_confirmatory") == 1
    assert "confirmatory_aggregate_seed" not in shell
    assert shell.index("run_stage editor_confirmatory") < shell.index(
        "--target editor_results"
    )
    assert shell.index("run_stage p2t_confirmatory") < shell.index(
        "--target p2t_results"
    )
    assert shell.index("--target editor_results") < shell.index("--target complete")


def test_launcher_binds_m0_final_results_and_nonblocking_robustness() -> None:
    shell = source()
    assert "M0_EVALUATION_READY.json" in shell
    assert "--target m0_evaluation" in shell
    assert "expected_result_artifacts" in shell
    assert "p2t_checkpoints" in shell
    assert "p2t_paired_rows" in shell
    assert "editor_checkpoints" in shell
    assert shell.index("run_stage validate_complete") < shell.index(
        "run_stage robustness_all_corpus"
    )
    assert "--strategy-mode all" in shell
    assert "does_not_invalidate_primary_route_a" in shell
    assert "SCRIBBLE_ROBUSTNESS_ALL_FAILED" in shell
    assert 'payload["test_access_receipt"] = receipt or None' in shell
    assert 'payload["run_root"] = run_root' in shell
    assert 'payload["evaluation_partition"] = evaluation_partition' in shell
    assert 'payload["test_access_receipt"]' in shell
    assert 'payload["frozen_checkpoint_bindings"]' in shell
    robustness_tail = shell[shell.index("ROBUSTNESS_ALL_EPISODES=") :]
    assert '"${TEST_ACCESS_RECEIPT}" "${RUN_ROOT}" "${EVALUATION_PARTITION}"' in robustness_tail


def test_pipeline_validator_revalidates_shared_receipt_and_rejects_val_receipts() -> None:
    validator = VALIDATOR.read_text(encoding="utf-8")
    assert "validate_consumed_receipt(" in validator
    assert 'evaluation_partition == "test"' in validator
    assert "validation pipeline must not carry a test-access receipt" in validator
    assert '"test_access_receipt_sha256": test_access_receipt_sha256' in validator
    assert "document.get(\"test_access_receipt_sha256\")" in validator


def test_launcher_preflights_env_v11_cc3d_and_official_sources() -> None:
    shell = source()
    assert 'receipt.get("schema_version") != "PETCT-NNUNET-ENV-v1.1"' in shell
    assert 'for key in ("cc3d_import", "cc3d_distribution_root")' in shell
    assert "official_autopetv_preflight" in shell
    assert 'OFFICIAL_RUNTIME_MANIFEST="${PETCT_AUTOPETV_RUNTIME_MANIFEST:-' in shell
    assert '"${OFFICIAL_RUNTIME_MANIFEST}"' in shell


def test_launcher_revalidates_f0_before_creating_the_run_root() -> None:
    shell = source()
    assert "orchestration/validate_petct_route_a_f0.py" in shell
    assert 'F0_READY="${PROJECT_ROOT}/route_a/manifests/F0_READY.json"' in shell
    assert 'CORE_ENV_MARKER="${EXP_ROOT}/envs/ENV_READY.done"' in shell
    assert (
        'EXPECTED_F0_ENV_BUNDLE="'
        '87a2261af9d99eb8232a078a2f7ba81cf9f3b4a6389410c296ca9b8671246006"'
        in shell
    )
    validation = shell.index('"${F0_VALIDATOR}" validate')
    first_run_root_mkdir = shell.index('mkdir -p "${RUN_ROOT}/logs"')
    first_scientific_stage = shell.index("run_stage environment_preflight")
    assert validation < first_run_root_mkdir < first_scientific_stage
    assert 'run_stage f0' not in shell
    assert '--official-runtime-manifest "${OFFICIAL_RUNTIME_MANIFEST}"' in shell


def test_launcher_binds_f0_receipt_into_primary_provenance_chain() -> None:
    shell = source()
    binding = (
        'payload["f0_readiness"] = {"path": f0_path, "sha256": f0_sha256, '
        '"bytes": int(f0_bytes)}'
    )
    assert binding in shell
    assert 'payload["f0_readiness"] = {"path":' in shell
    assert '"${F0_READY_PATH}" "${F0_READY_SHA256}"' in shell
    assert '"${F0_READY_BYTES}"' in shell
    # Later P2T/editor/final input documents copy their validated base payload,
    # so the same F0 record remains inside the final hash-bound inputs.
    assert shell.count('payload = json.load(open(base, encoding="utf-8"))') >= 3


def test_launcher_wires_native_residual_and_scribble_leaf_receipts() -> None:
    shell = source()
    assert 'RESIDUAL_READY="${RUN_ROOT}/artifacts/RESIDUAL_READY.json"' in shell
    assert 'CONTROLLED_DATA_READY="${RUN_ROOT}/artifacts/CONTROLLED_DATA_READY.json"' in shell
    assert (
        'NATURAL_PRIMARY_DATA_READY="${RUN_ROOT}/artifacts/'
        'NATURAL_PRIMARY_DATA_READY.json"' in shell
    )
    assert (
        'NATURAL_ROBUSTNESS_DATA_READY="${RUN_ROOT}/artifacts/'
        'NATURAL_ROBUSTNESS_DATA_READY.json"' in shell
    )
    residual = shell[shell.index("run_stage residual_manifest") : shell.index("CONTROLLED_EPISODES=")]
    controlled = shell[shell.index("CONTROLLED_BUILD=(") : shell.index("run_stage controlled_matched_states")]
    natural = shell[shell.index("run_stage natural_autopetv_scribbles") : shell.index("CONTROLLED_TENSORS=")]
    robustness = shell[shell.index("if run_stage robustness_all_corpus") : shell.index("ROBUSTNESS_STATUS=0", shell.index("if run_stage robustness_all_corpus"))]
    assert '--ready-receipt "${RESIDUAL_READY}"' in residual
    assert '--ready-receipt "${CONTROLLED_DATA_READY}"' in controlled
    assert '--official-runtime-manifest "${OFFICIAL_RUNTIME_MANIFEST}"' in controlled
    assert '--residual-ready "${RESIDUAL_READY}"' in natural
    assert '--ready-receipt "${NATURAL_PRIMARY_DATA_READY}"' in natural
    assert '--official-runtime-manifest "${OFFICIAL_RUNTIME_MANIFEST}"' in natural
    assert '--residual-ready "${RESIDUAL_READY}"' in robustness
    assert '--ready-receipt "${NATURAL_ROBUSTNESS_DATA_READY}"' in robustness
    assert '--official-runtime-manifest "${OFFICIAL_RUNTIME_MANIFEST}"' in robustness
    assert 'SCOPED_RESIDUALS="${RESIDUAL_MANIFEST}"' in shell
    assert "run_stage scope_natural_residuals" not in shell
    assert 'payload["residual_ready"]' in shell
    assert 'payload["controlled_data_ready"]' in shell
    assert 'payload["natural_primary_data_ready"]' in shell
    assert 'payload["natural_robustness_data_ready"]' in shell


def test_tensor_materialization_revalidates_the_frozen_learning_split() -> None:
    shell = source()
    calls = [
        line_index
        for line_index, line in enumerate(shell.splitlines())
        if "materialize_petct_learning_tensors.py" in line
        and '"${PYTHON}"' in line
    ]
    assert len(calls) == 2
    lines = shell.splitlines()
    for index in calls:
        invocation = "\n".join(lines[index : index + 9])
        assert '--learning-split "${LEARNING_SPLIT}"' in invocation
        assert '--partitions "${SELECTED_PARTITIONS[@]}"' in invocation
        assert '"${TEST_ACCESS_ARGS[@]}"' in invocation
    assert '"simulate_scribble_from_label"' in shell
    assert '"MetricEvaluator"' in shell


def test_formal_test_is_evaluation_only_and_uses_exact_freeze_bindings() -> None:
    shell = source()
    assert "export-checkpoints" in shell
    assert "FROZEN_CHECKPOINT_BINDINGS.json" in shell
    assert "selected_checkpoint:p2t:%s:%s:seed%s" in shell
    assert "selected_checkpoint:editor:%s:%s:seed%s" in shell
    assert "training_manifest.path" in shell
    assert '--training-manifest "${training_manifest}"' in shell
    for guarded in (
        'if [[ "${EVALUATION_PARTITION}" == "val" ]]; then\n  run_p2t_train_queue',
        'if [[ "${EVALUATION_PARTITION}" == "val" ]]; then\n  run_editor_train_queue',
    ):
        assert guarded in shell
    for invocation in (
        'run_p2t_train_queue "${GPU0_ID}" 0 &',
        'run_editor_train_queue "${GPU0_ID}" 0 &',
    ):
        offset = shell.index(invocation)
        guard = shell.rfind(
            'if [[ "${EVALUATION_PARTITION}" == "val" ]]; then', 0, offset
        )
        assert guard >= 0
        assert shell.find("\nfi", offset) >= 0
    assert 'if [[ "${EVALUATION_PARTITION}" == "test" ]]; then\n  run_' not in shell


def test_formal_test_uses_freeze_bound_oof_environment_and_m0_receipts() -> None:
    shell = source()
    for role in (
        "m0_oof_receipt",
        "m0_validation_receipt",
        "environment_receipt",
    ):
        assert f"--role {role}" in shell
    assert (
        'if [[ "${EVALUATION_PARTITION}" == "val" && ! -e "${OOF_READY}" ]]; then'
        in shell
    )
    assert 'if [[ ! -e "${OOF_READY}" ]]; then' not in shell
    assert 'ready.get("validated_bundle", {}).get("splits_final", {})' in shell
    assert 'payload["frozen_m0_validation_receipt"]' in shell
    assert 'payload["frozen_environment_receipt"]' in shell
    assert 'payload["frozen_oof_receipt"]' in shell
    for token in (
        '--role environment_receipt)',
        '--role m0_oof_receipt)',
        '--role m0_validation_receipt)',
    ):
        assert shell.index(token) < shell.index("run_stage environment_preflight")
