from __future__ import annotations

import re
import sys
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
SCRIPTS = PROJECT / "scripts"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(SCRIPTS / "baseline"))

from common.petct_route_a_core import plan_gpu_queues  # noqa: E402


SCRIPT = SCRIPTS / "baseline" / "launch_petct_m0_parallel_continuation.sh"


def _source() -> str:
    return SCRIPT.read_text(encoding="utf-8")


def test_continuation_plan_keeps_two_serial_queues_while_fold0_runs() -> None:
    plan = plan_gpu_queues(running_folds=[0], gpu_for_running={0: 0})

    assert plan["active"] == {"0": 0, "1": None}
    assert plan["queues"] == {"0": [2, 4], "1": [1, 3]}
    assert plan["launch_performed"] is False


def test_shell_is_plan_only_until_two_part_execution_gate() -> None:
    source = _source()

    assert "status: PLAN_ONLY" in source
    assert "launch_performed: false" in source
    assert "--execute" in source
    assert "--confirm" in source
    assert "EXECUTE_PETCT_M0_CONTINUATION_${CAMPAIGN_ID}" in source
    gate_end = source.index('if [[ "${CONFIRM_TOKEN}" != "${EXPECTED_TOKEN}" ]]')
    environment_source = source.index('source "${SCRIPT_DIR}/../common/petct_m0_common.sh"')
    assert environment_source > gate_end
    assert source.index('if [[ "${EXECUTE}" != "true" ]]') < environment_source


def test_shell_never_relaunches_fold0_and_starts_gpu1_queue_independently() -> None:
    source = _source()

    gpu1 = re.search(r"run_gpu1_queue\(\) \{(?P<body>.*?)\n\}", source, re.S)
    gpu0 = re.search(r"run_gpu0_queue\(\) \{(?P<body>.*?)\n\}", source, re.S)
    assert gpu1 is not None and gpu0 is not None
    assert re.findall(r"run_fold_to_verified (\d)", gpu1.group("body")) == ["1", "3"]
    assert re.findall(r"run_fold_to_verified (\d)", gpu0.group("body")) == ["2", "4"]
    assert "wait_for_verified_fold 0" in gpu0.group("body")
    assert "run_fold_to_verified 0" not in source
    assert source.index("run_gpu1_queue &") < source.index("run_gpu0_queue &")


def test_shell_uses_locks_fold_actions_and_verified_publish_barrier() -> None:
    source = _source()

    assert "m0_parallel_continuation.lock" in source
    assert "flock -n 9" in source
    assert "fold_${fold}.lock" in source
    assert "fold-action" in source
    assert "SKIP_VERIFIED" in source
    assert "idempotent skip" in source
    assert "for fold in 0 1 2 3 4" in source
    assert source.index("for fold in 0 1 2 3 4") < source.index(
        '"${PYTHON}" "${VALIDATOR}" publish-full-ready'
    )
    assert "FULL_TRAIN_READY will not be published" in source


def test_shell_waits_for_both_parallel_workers_before_publication() -> None:
    source = _source()

    assert 'wait "${GPU1_WORKER}"' in source
    assert 'wait "${GPU0_WORKER}"' in source
    publish = source.index('"${PYTHON}" "${VALIDATOR}" publish-full-ready')
    assert source.index('wait "${GPU1_WORKER}"') < publish
    assert source.index('wait "${GPU0_WORKER}"') < publish
    assert "GPU0_RC" in source and "GPU1_RC" in source
