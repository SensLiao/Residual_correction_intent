from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "scripts"))

from common.petct_learning import (  # noqa: E402
    LearningContractError,
    validate_manifest_rows_against_frozen_learning_split,
)


def _split(root: Path) -> Path:
    path = root / "learning-split.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "PETCT-LEARNING-SPLIT-v1.0",
                "status": "FROZEN_BEFORE_MODEL_SELECTION",
                "patient_count": 2,
                "case_count": 2,
                "case_counts": {"train": 1, "val": 0, "test": 1},
                "patients": [
                    {
                        "patient_id": "train-patient",
                        "partition": "train",
                        "case_ids": ["train-case"],
                    },
                    {
                        "patient_id": "test-patient",
                        "partition": "test",
                        "case_ids": ["test-case"],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_test_case_relabelled_as_train_is_rejected_without_opening_tensor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    split = _split(tmp_path)
    tensor = tmp_path / "must-not-be-opened.npz"
    original_open = Path.open

    def guarded_open(path: Path, *args, **kwargs):
        if Path(path) == tensor:
            raise AssertionError("tensor was opened before split validation")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    rows = [
        {
            "case_id": "test-case",
            "patient_id": "test-patient",
            "partition": "train",
            "episode_id": "episode-1",
            "learning_split_sha256": _sha(split),
            "visible_npz": str(tensor),
        }
    ]
    with pytest.raises(LearningContractError, match="partition differs"):
        validate_manifest_rows_against_frozen_learning_split(
            rows,
            split,
            require_episode_id=True,
            allowed_partitions={"train", "val", "test"},
        )


def test_case_patient_and_split_hash_are_both_independently_checked(tmp_path: Path) -> None:
    split = _split(tmp_path)
    base = {
        "case_id": "train-case",
        "patient_id": "wrong-patient",
        "partition": "train",
        "episode_id": "episode-1",
        "learning_split_sha256": _sha(split),
    }
    with pytest.raises(LearningContractError, match="patient differs"):
        validate_manifest_rows_against_frozen_learning_split(
            [base],
            split,
            require_episode_id=True,
            allowed_partitions={"train"},
        )
    with pytest.raises(LearningContractError, match="split hash"):
        validate_manifest_rows_against_frozen_learning_split(
            [{**base, "patient_id": "train-patient", "learning_split_sha256": "0" * 64}],
            split,
            require_episode_id=True,
            allowed_partitions={"train"},
        )


def test_owned_tensor_leafs_guard_manifest_before_dataset_or_np_load() -> None:
    paths_and_sinks = {
        "scripts/p2t/train_petct_p2t.py": "EpisodeDataset(",
        "scripts/editor/train_petct_residual_editor.py": "EpisodeDataset(",
        "scripts/editor/infer_petct_residual_editor.py": "torch.load(args.checkpoint",
        "scripts/evaluation/evaluate_petct_correction.py": 'with np.load(row["prediction_npz"]',
    }
    for relative, sink in paths_and_sinks.items():
        source = (PROJECT / relative).read_text(encoding="utf-8")
        assert "validate_manifest_rows_against_frozen_learning_split(" in source
        assert source.index("validate_manifest_rows_against_frozen_learning_split(") < source.index(sink)
