from __future__ import annotations

from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT / "scripts" / "baseline" / "run_petct_m0_oof_parallel.sh"


def _source() -> str:
    return SCRIPT.read_text(encoding="utf-8")


def test_parallel_oof_uses_two_serial_fold_queues() -> None:
    source = _source()

    assert "GPU0_FOLDS=(0 2 4)" in source
    assert "GPU1_FOLDS=(1 3)" in source
    assert 'run_fold_queue "${GPU0_ID}" "${GPU0_FOLDS[@]}" &' in source
    assert 'run_fold_queue "${GPU1_ID}" "${GPU1_FOLDS[@]}" &' in source
    assert 'export CUDA_VISIBLE_DEVICES="${gpu_id}"' in source
    assert '--device cuda:0 --source-root "${NNUNET_SOURCE}"' in source


def test_parallel_oof_prefers_receipt_bound_actual_validation_handoff() -> None:
    source = _source()

    handoff_gate = source.index(
        'if [[ "${OOF_HANDOFF_AVAILABLE}" == "true" ]]'
    )
    handoff = source.index("--from-actual-validation", handoff_gate)
    dedicated = source.index('GPU0_FOLDS=(0 2 4)', handoff)
    validate = source.index('"${VALIDATOR}" validate-oof')
    assert handoff_gate < handoff < dedicated < validate
    assert 'ready.get("oof_handoff_inputs_present") is True' in source
    assert 'ready.get("actual_inference_gate_required") is False' in source


def test_parallel_oof_fails_before_publish_and_revalidates_all_receipts() -> None:
    source = _source()

    wait_gpu0 = source.index('wait "${GPU0_PID}"')
    wait_gpu1 = source.index('wait "${GPU1_PID}"')
    failure_gate = source.index("if [[ ${GPU0_STATUS} -ne 0 || ${GPU1_STATUS} -ne 0 ]]")
    validate = source.index('"${VALIDATOR}" validate-oof')
    commit = source.index('"${VALIDATOR}" commit-run')
    publish = source.index('"${VALIDATOR}" publish')
    assert wait_gpu0 < wait_gpu1 < failure_gate < validate < commit < publish
    assert "OOF_READY already exists; refusing overwrite" in source
    assert "no OOF_READY was published" in source


def test_parallel_oof_bounds_host_threads_and_defaults_to_gpu_zero_one() -> None:
    source = _source()

    assert 'GPU0_ID="${1:-0}"' in source
    assert 'GPU1_ID="${2:-1}"' in source
    assert 'if [[ "${GPU0_ID}" == "${GPU1_ID}" ]]' in source
    assert "export OMP_NUM_THREADS=1" in source
    assert "export MKL_NUM_THREADS=1" in source
    assert "export OPENBLAS_NUM_THREADS=1" in source
    assert "export NUMEXPR_NUM_THREADS=1" in source
