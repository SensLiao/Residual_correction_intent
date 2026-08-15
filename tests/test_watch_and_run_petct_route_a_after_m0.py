from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
SCRIPT = (
    PROJECT
    / "scripts"
    / "orchestration"
    / "watch_and_run_petct_route_a_after_m0.sh"
)


def source() -> str:
    return SCRIPT.read_text(encoding="utf-8")


def test_watcher_pins_exact_campaign_paths_and_bounded_polling() -> None:
    shell = source()
    assert "set -euo pipefail" in shell
    assert 'DEFAULT_CAMPAIGN_ID="petct_m0_5fold_20260717T121633Z"' in shell
    assert "/route_a/runs/route_a_val_${DEFAULT_CAMPAIGN_ID}" in shell
    assert "/route_a/external/nninteractive_val_${DEFAULT_CAMPAIGN_ID}" in shell
    assert (
        "nnunet/audits/psma_v3_nifti_audit_20260717T150905/"
        "psma_v3_nifti_audit.json" in shell
    )
    assert 'FULL_TRAIN_READY="${FULL_TRAIN_READY:-${PROJECT_ROOT}/nnunet/manifests/FULL_TRAIN_READY.json}"' in shell
    assert "POLL_SECONDS=120" in shell
    assert "MAX_WAIT_SECONDS=604800" in shell
    assert "now + POLL_SECONDS > WAIT_DEADLINE" in shell
    assert 'sleep "${POLL_SECONDS}"' in shell


def test_watcher_revalidates_full_ready_and_every_fold_receipt() -> None:
    shell = source()
    assert '[[ ! -f "${FULL_TRAIN_READY}" ]]' in shell
    assert '[[ -L "${FULL_TRAIN_READY}" ]]' in shell
    assert '"status": "COMMITTED"' in shell
    assert '"full_training_status": "PASS"' in shell
    assert '"folds_completed": [0, 1, 2, 3, 4]' in shell
    assert 'validate-campaign "${CAMPAIGN_ROOT}"' in shell
    assert "for fold in 0 1 2 3 4; do" in shell
    assert 'fold-action "${CAMPAIGN_ROOT}" "${fold}"' in shell
    assert '[[ "${action}" != "SKIP_VERIFIED" ]]' in shell


def test_watcher_issues_once_and_revalidates_exact_f0_before_gpu_polling() -> None:
    shell = source()
    assert 'F0_READY="${PROJECT_ROOT}/route_a/manifests/F0_READY.json"' in shell
    assert 'F0_VALIDATOR="${PROJECT_ROOT}/scripts/orchestration/validate_petct_route_a_f0.py"' in shell
    assert 'CORE_ENV_MARKER="${PROJECT_ROOT}/nnunet/envs/ENV_READY.done"' in shell
    assert (
        'EXPECTED_F0_ENV_BUNDLE="'
        '87a2261af9d99eb8232a078a2f7ba81cf9f3b4a6389410c296ca9b8671246006"'
        in shell
    )
    assert 'if [[ ! -e "${F0_READY}" && ! -L "${F0_READY}" ]]; then' in shell
    assert '"${CORE_PYTHON}" "${F0_VALIDATOR}" issue' in shell
    assert '--output "${F0_READY}" --test-log "${F0_TEST_LOG}"' in shell
    assert '--official-runtime-manifest "${OFFICIAL_RUNTIME_MANIFEST}"' in shell
    assert '[[ -L "${F0_READY}" ]]' in shell
    f0_issue = shell.index('"${CORE_PYTHON}" "${F0_VALIDATOR}" issue')
    f0_validation = shell.index('"${CORE_PYTHON}" "${F0_VALIDATOR}" validate')
    gpu_polling = shell.index("gpus_are_idle()")
    route_launch = shell.index(
        '"${ROUTE_LAUNCHER}" --run-root "${ROUTE_RUN_ROOT}" --partition val'
    )
    assert f0_issue < f0_validation < gpu_polling < route_launch


def test_watcher_only_observes_gpu_processes_and_never_kills() -> None:
    shell = source()
    assert "nvidia-smi" in shell
    assert "--query-compute-apps=pid" in shell
    assert 'for gpu in "${GPU0_ID}" "${GPU1_ID}"' in shell
    assert "Waiting for both selected GPUs to have no compute process" in shell
    lowered = shell.lower()
    assert "kill " not in lowered
    assert "pkill" not in lowered
    assert "killall" not in lowered


def test_watcher_verifies_inputs_and_caps_cpu_threads() -> None:
    shell = source()
    assert 'CORE_PYTHON_RESOLVED="$(readlink -f -- "${CORE_PYTHON}"' in shell
    assert '"${CORE_PYTHON_RESOLVED}" != "${CORE_ENV_PREFIX}"/bin/python*' in shell
    assert "PSMA audit is not a clean PASS" in shell
    assert "a2124e8aa4207e53ac93259214a35b7cf74626f83ab164e519769f86557d7cd2" in shell
    assert "93e303219deb46b10fc5e5532873a42745aec1ecd6f78335f36cebba62104b83" in shell
    assert 'export PETCT_PSMA_AUDIT_JSON="${AUDIT_JSON}"' in shell
    assert 'export PETCT_AUTOPETV_SIMULATOR="${OFFICIAL_SIMULATOR}"' in shell
    assert 'export PETCT_AUTOPETV_METRICS="${OFFICIAL_METRICS}"' in shell
    for variable in (
        "OMP_NUM_THREADS=4",
        "MKL_NUM_THREADS=4",
        "OPENBLAS_NUM_THREADS=4",
        "NUMEXPR_NUM_THREADS=4",
        "TORCH_NUM_THREADS=4",
    ):
        assert variable in shell
    assert "torch.get_num_threads() > 4" in shell


def test_watcher_does_not_precreate_or_reuse_run_roots() -> None:
    shell = source()
    assert '[[ -e "${ROUTE_RUN_ROOT}" || -L "${ROUTE_RUN_ROOT}" ]]' in shell
    assert '[[ -e "${EXTERNAL_RUN_ROOT}" || -L "${EXTERNAL_RUN_ROOT}" ]]' in shell
    assert shell.count("validate_fresh_run_roots") >= 3
    assert "mkdir -p" not in shell
    assert "run roots must remain below the project root" in shell


def test_watcher_runs_val_route_foreground_then_nninteractive_on_gpu1() -> None:
    shell = source()
    route_call = (
        '"${ROUTE_LAUNCHER}" --run-root "${ROUTE_RUN_ROOT}" --partition val'
    )
    external_call = (
        '"${EXTERNAL_LAUNCHER}" --route-a-run-root "${ROUTE_RUN_ROOT}"'
    )
    assert route_call in shell
    assert '--gpu0 "${GPU0_ID}" --gpu1 "${GPU1_ID}"' in shell
    assert external_call in shell
    assert '--run-root "${EXTERNAL_RUN_ROOT}" --methods nninteractive --partition val' in shell
    assert '--gpu-nninteractive "${GPU1_ID}"' in shell
    assert "--allow-test-access" not in shell
    assert shell.index(route_call) < shell.index("ROUTE_A_COMPLETE.json")
    receipt_offset = shell.index("ROUTE_A_COMPLETE.json")
    rebind_offset = shell.index("PETCT_SKIP_INSTALL=1 PIP_NO_INDEX=1")
    external_offset = shell.index(external_call)
    assert receipt_offset < rebind_offset < external_offset
    assert 'PETCT_NNINTERACTIVE_SMOKE_GPU="${GPU1_ID}"' in shell
    assert '"setup_mode": "VERIFY_EXISTING_NO_INSTALL"' in shell
    route_line = next(line for line in shell.splitlines() if route_call in line)
    assert not route_line.rstrip().endswith("&")


def test_external_failure_is_recorded_without_mutating_primary_receipt() -> None:
    shell = source()
    assert "PETCT-ROUTE-A-PIPELINE-RECEIPT-v2.0" in shell
    assert "PETCT-ROUTE-A-PIPELINE-RECEIPT-v1.0" not in shell
    assert "ROUTE_COMPLETE_SHA256=" in shell
    assert "ROUTE_COMPLETE_SHA256_AFTER=" in shell
    assert "PETCT-POSTBASELINE-FAILURE-v1.0" in shell
    assert "EXTERNAL_NNINTERACTIVE_FAILED.json" in shell
    assert '"primary_receipt_preserved": preserved == "true"' in shell
    assert 'exit "${EXTERNAL_STATUS}"' in shell
    assert "failure receipt recorded without editing ROUTE_A_COMPLETE" in shell
    assert 'receipt.get("selected_methods") != ["nninteractive"]' in shell


def test_rebind_failure_is_separate_and_cannot_invalidate_primary_route() -> None:
    shell = source()
    assert "NNINTERACTIVE_ENVIRONMENT_REBIND_FAILED.json" in shell
    assert "nninteractive_environment_rebind" in shell
    assert "primary Route A receipt was not edited" in shell
    assert shell.index("ROUTE_COMPLETE_SHA256=") < shell.index(
        "PETCT_SKIP_INSTALL=1 PIP_NO_INDEX=1"
    )
    assert shell.index("NNINTERACTIVE_ENVIRONMENT_REBIND_FAILED.json") < shell.index(
        '"${EXTERNAL_LAUNCHER}" --route-a-run-root'
    )
    initial_prerequisites = shell[
        shell.index('for required in "${FULL_TRAIN_VALIDATOR}"') : shell.index(
            "validate_fresh_run_roots()"
        )
    ]
    assert "NNINTERACTIVE_SETUP" not in initial_prerequisites
    assert "EXTERNAL_LAUNCHER" not in initial_prerequisites
