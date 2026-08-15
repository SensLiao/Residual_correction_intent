from __future__ import annotations

from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
SCRIPT = (
    PROJECT
    / "scripts"
    / "orchestration"
    / "run_petct_external_comparators_after_data.sh"
)


def source() -> str:
    return SCRIPT.read_text(encoding="utf-8")


def test_v2_campaign_fails_closed_before_mkdir_or_gpu_for_remove_unsupported() -> None:
    shell = source()
    gate = shell.index('"status": "REMOVE_UNSUPPORTED"')
    mkdir = shell.index('mkdir -p "${RUN_ROOT}/logs"')
    first_gpu = shell.index("CUDA_VISIBLE_DEVICES=", mkdir)
    assert gate < mkdir < first_gpu
    assert '"union_with_m0_role": "LEGACY_V1_ADD_DERIVED_ONLY"' in shell
    assert '"gpu_started": False' in shell
    assert "raise SystemExit(42)" in shell


def test_launcher_gates_oof_and_ready_environments_before_any_adapter() -> None:
    shell = source()
    oof_gate = shell.index('validate-ready-receipt "${OOF_READY}"')
    mkdir = shell.index('mkdir -p "${RUN_ROOT}/logs"')
    scribble_adapter = shell.index("scribbleprompt_petct_adapter.py", mkdir)
    nninteractive_adapter = shell.index('"${NNI_PYTHON}" "${NNI_ADAPTER}"', mkdir)

    assert 'get("state") != "ARGV_WIRED"' in shell
    assert "load_and_validate_contract(config_path)" in shell
    assert "validate_execution_admission(" in shell
    assert "PETCT-SCRIBBLEPROMPT-ENV-v1.0" in shell
    assert "PETCT-NNINTERACTIVE-ENV-v1.1" in shell
    assert 'receipt.get("status") == "PASS"' in shell
    assert 'receipt.get("model_load_smoke") == "PASS"' in shell
    assert 'receipt.get("initial_m0_api_smoke") == "PASS"' in shell
    assert 'receipt.get("scribble_api_smoke") == "PASS"' in shell
    assert 'receipt.get("adapter_cli_smoke") == "PASS"' in shell
    assert 'receipt.get("source_bundle_sha256") == source_hash' in shell
    assert 'receipt.get("config_sha256") == sha256(config_path)' in shell
    assert 'receipt.get("environment_freeze_sha256") == sha256(nni_environment_freeze)' in shell
    assert "for method_id in selected:" in shell
    assert 'execution.get("network_policy") != "NO_DOWNLOADS"' in shell
    assert oof_gate < mkdir < scribble_adapter
    assert oof_gate < mkdir < nninteractive_adapter
    assert "--target editor_data" in shell


def test_launcher_requires_an_explicit_method_subset_with_no_implicit_default() -> None:
    shell = source()

    assert "--methods ID[,ID]" in shell
    assert 'METHODS_RAW=""' in shell
    assert '|| -z "${METHODS_RAW}"' in shell
    assert 'Unsupported method in --methods: ${method:-<empty>}' in shell
    assert "--methods must select at least one supported comparator" in shell
    assert "for method_id in selected:" in shell
    assert "for method_id, selection in expected.items():" not in shell


def test_launcher_scopes_prerequisites_and_queues_to_selected_methods() -> None:
    shell = source()
    common_start = shell.index("declare -a REQUIRED_FILES=(")
    common_end = shell.index("declare -a SELECTED_PYTHONS=", common_start)
    common_prerequisites = shell[common_start:common_end]

    assert '${SP_PYTHON}' not in common_prerequisites
    assert '${NNI_PYTHON}' not in common_prerequisites
    assert '${SP_ENV_RECEIPT}' not in common_prerequisites
    assert '${NNI_ENV_RECEIPT}' not in common_prerequisites
    assert 'if [[ ${SELECT_SP} -eq 1 ]]' in shell
    assert 'if [[ ${SELECT_NNI} -eq 1 ]]' in shell


def test_launcher_preserves_per_method_policy_tables_without_pooling() -> None:
    shell = source()

    assert "for policy in union_with_m0 native_slice_replace" not in shell
    assert "for policy in union_with_m0 native_full_mask" not in shell
    assert 'run_stage "infer_scribbleprompt_once"' in shell
    assert 'run_stage "infer_nninteractive_once"' in shell
    assert '--output-policy native_slice_replace' in shell
    assert '--output-policy native_full_mask' in shell
    assert '--derived-union-output-manifest "${union_manifest}"' in shell
    assert '--derived-union-output-dir "${union_dir}"' in shell
    assert 'evaluate_one scribbleprompt union_with_m0 "${union_manifest}"' in shell
    assert 'evaluate_one scribbleprompt native_slice_replace "${native_manifest}"' in shell
    assert 'evaluate_one nninteractive union_with_m0 "${union_manifest}"' in shell
    assert 'evaluate_one nninteractive native_full_mask "${native_manifest}"' in shell
    assert "run_scribbleprompt_queue &" in shell
    assert "run_nninteractive_queue &" in shell
    assert 'QUEUE_NAMES+=("scribbleprompt")' in shell
    assert 'QUEUE_NAMES+=("nninteractive")' in shell
    assert 'CUDA_VISIBLE_DEVICES="${GPU_SP}"' in shell
    assert 'CUDA_VISIBLE_DEVICES="${GPU_NNI}"' in shell
    assert "finalize_petct_external_comparators.py" in shell
    assert "COMPLETE_ARGS" in shell
    assert "PETCT-EXTERNAL-COMPARATORS-COMPLETE-v1.2" in (
        PROJECT
        / "scripts"
        / "comparators"
        / "finalize_petct_external_comparators.py"
    ).read_text(encoding="utf-8")
    assert '"cross_dimensional_pooling": "FORBIDDEN"' not in shell  # read from each summary
    assert "EXTERNAL_COMPARATORS_COMPLETE.json" in shell


def test_launcher_has_explicit_independent_test_gates_and_no_download_mode() -> None:
    shell = source()

    assert "--allow-test" not in shell
    assert "--confirm-test-access" not in shell
    assert "--test-access-receipt" in shell
    assert 'petct_test_access.py" validate' in shell
    assert '--run-root "${ROUTE_A_RUN_ROOT}"' in shell
    assert "never creates or" in shell
    assert "consumes test authorization" in shell
    assert "must be a child of the receipt-bound Route-A root" in shell
    assert "HF_HUB_OFFLINE=1" in shell
    assert "TRANSFORMERS_OFFLINE=1" in shell
    assert "PIP_NO_INDEX=1" in shell


def test_formal_test_resolves_external_freeze_before_any_route_a_data_or_gpu() -> None:
    shell = source()
    freeze_gate = shell.index("resolve-test-external")
    pipeline_read = shell.index("readarray -t DATA_PATHS")
    oof_read = shell.index('validate-ready-receipt "${OOF_READY}"')
    gpu = shell.index('run_stage "infer_nninteractive_once"')

    assert freeze_gate < pipeline_read < oof_read < gpu
    assert "receipt-bound final freeze does not contain nninteractive" in (
        PROJECT / "scripts" / "common" / "petct_development_freeze.py"
    ).read_text(encoding="utf-8")
    assert "Formal test external execution is admitted only for the frozen nninteractive role" in shell
    assert '--frozen-external-admission "${FROZEN_EXTERNAL_ADMISSION}"' in shell
    assert '"${NNI_PYTHON}" "${NNI_ADAPTER}"' in shell
    assert '--model-folder "${NNI_MODEL}"' in shell


def test_formal_test_bootstrap_and_runtime_paths_are_freeze_bound() -> None:
    shell = source()

    assert (
        'BOOTSTRAP_CORE_PYTHON="${PROJECT_ROOT}/envs/petct_nnunet_v281/bin/python"'
        in shell
    )
    assert '"${BOOTSTRAP_CORE_PYTHON}" -I "${PROJECT_ROOT}/scripts/common/' in shell
    assert '"${BOOTSTRAP_CORE_PYTHON}" -I -c' in shell
    assert 'p["nninteractive_python"]["path"]' in shell
    assert 'p["core_python"]["path"]' in shell
    assert 'p["official_metrics"]["path"]' in shell
    assert "Isolated bootstrap Python differs from the receipt-bound core Python" in shell
    assert "Formal test rejects ${override_name} override outside the receipt-bound final freeze" in shell
    assert '--core-python "${CORE_PYTHON}"' in shell
    assert '--official-metrics "${OFFICIAL_METRICS}"' in shell
    assert '--nninteractive-python "${NNI_PYTHON}"' in shell
    assert '--natural-episode-manifest "${NATURAL_EPISODES}"' in shell
    assert '--route-a-run-root "${ROUTE_A_RUN_ROOT}"' in shell
