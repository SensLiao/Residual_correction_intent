from __future__ import annotations

import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import torch


PROJECT = Path(__file__).resolve().parents[1]
SCRIPTS = PROJECT / "scripts"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(SCRIPTS / "data"))

import materialize_petct_learning_tensors as tensor_materializer  # noqa: E402
from materialize_petct_learning_tensors import (  # noqa: E402
    axial_stack,
    main,
    normalize_ct,
    normalize_pet,
    physical_crop_resample_2d,
    scribble_mask,
)
from common.petct_learning import (  # noqa: E402
    EpisodeDataset,
    LearningContractError,
    sha256_file,
)
from common.petct_models import TARGET_TO_ID, SCOPE_TO_ID  # noqa: E402


def test_normalization_and_axial_edge_replication() -> None:
    volume = np.arange(3 * 4 * 2, dtype=np.float32).reshape(3, 4, 2)
    assert axial_stack(volume, 0).shape == (5, 3, 4)
    assert np.array_equal(axial_stack(volume, 0)[0], volume[:, :, 0])
    assert normalize_ct(np.array([-2000, 0, 2000])) .tolist() == [-1.0, 0.0, 1.0]
    pet, stats = normalize_pet(np.array([0.0, 1.0, 3.0, 7.0]))
    assert np.isfinite(pet).all() and stats["log_iqr"] > 0
    with pytest.raises(ValueError, match="NaN or Inf"):
        normalize_ct(np.asarray([np.nan]))
    with pytest.raises(ValueError, match="NaN or Inf"):
        normalize_pet(np.asarray([np.inf]))


def test_scribble_mask_requires_valid_coordinates() -> None:
    mask = scribble_mask((4, 4, 3), [[1, 2, 1]])
    assert mask.sum() == 1
    with pytest.raises(ValueError, match="out of bounds"):
        scribble_mask((4, 4, 3), [[4, 0, 0]])


def test_physical_crop_makes_mixed_source_shapes_batchable() -> None:
    first = np.zeros((73, 91), dtype=np.float32)
    second = np.zeros((128, 64), dtype=np.float32)
    first[36, 45] = 1
    second[64, 32] = 1
    out_a = physical_crop_resample_2d(
        first,
        center_xy=(36, 45),
        spacing_xy=(2.0, 2.0),
        field_mm=192.0,
        output_size=256,
        order=0,
    )
    out_b = physical_crop_resample_2d(
        second,
        center_xy=(64, 32),
        spacing_xy=(1.5, 2.5),
        field_mm=192.0,
        output_size=256,
        order=0,
    )
    assert out_a.shape == out_b.shape == (256, 256)
    assert out_a.any() and out_b.any()


def test_inference_dataset_never_opens_evaluation_truth(tmp_path: Path) -> None:
    visible = tmp_path / "visible.npz"
    scribble = np.zeros((8, 8), np.uint8)
    scribble[3, 3] = 1
    np.savez_compressed(visible, visual=np.zeros((17, 8, 8), np.float32), m0=np.zeros((8, 8), np.uint8), scribble=scribble, cue_fg=scribble, cue_bg=np.zeros_like(scribble), spacing_xy=np.ones(2))
    missing_eval = tmp_path / "missing-eval.npz"
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps({"episode_id":"ep","patient_id":"p","partition":"test","goal":"ADD_SAME_LOCAL","operation":"ADD","target":"SAME","scope":"LOCAL","visible_npz":str(visible),"visible_sha256":sha256_file(visible),"evaluation_npz":str(missing_eval),"evaluation_sha256":"missing","geometry":{"output_spacing_xy":[1.0,1.0]}}) + "\n",
        encoding="utf-8",
    )
    dataset = EpisodeDataset(manifest, "test", load_evaluation=False)
    assert dataset[0]["visual"].shape == (17, 8, 8)
    with pytest.raises(LearningContractError, match="missing evaluation_npz"):
        EpisodeDataset(manifest, "test", load_evaluation=True)[0]


def test_visual_state_only_keeps_pet_ct_m0_but_zeros_model_scribble_and_intent(
    tmp_path: Path,
) -> None:
    visible = tmp_path / "visible.npz"
    visual = np.zeros((17, 8, 8), np.float32)
    visual[:5] = 2.0
    visual[5:10] = 3.0
    visual[12] = 1.0
    visual[15, 3, 3] = 1.0
    np.savez_compressed(
        visible,
        visual=visual,
        m0=np.ones((8, 8), np.uint8),
        scribble=visual[15].astype(np.uint8),
        cue_fg=visual[15].astype(np.uint8),
        cue_bg=visual[16].astype(np.uint8),
        spacing_xy=np.ones(2, np.float32),
    )
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "episode_id": "ep",
                "patient_id": "p",
                "partition": "test",
                "goal": "ADD_SAME_LOCAL",
                "operation": "ADD",
                "target": "SAME",
                "scope": "LOCAL",
                "visible_npz": str(visible),
                "visible_sha256": sha256_file(visible),
                "evaluation_npz": str(tmp_path / "unused.npz"),
                "evaluation_sha256": "unused",
                "geometry": {"output_spacing_xy": [1.0, 1.0]},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    sample = EpisodeDataset(
        manifest,
        "test",
        editor_condition="visual_state_only",
        load_evaluation=False,
    )[0]
    assert sample["visual"].shape == (12, 8, 8)
    assert torch.all(sample["visual"][:5] == 2.0)
    assert torch.all(sample["visual"][5:10] == 3.0)
    assert torch.all(sample["visual"][10] == 1.0)
    assert torch.count_nonzero(sample["visual"][11]) == 0
    assert sample["target_id"].item() == TARGET_TO_ID["NULL"]
    assert sample["scope_id"].item() == SCOPE_TO_ID["NULL"]
    # The model-facing scribble channel is zero, while the immutable raw prompt
    # remains available separately for prompt-relative evaluation metrics.
    assert torch.count_nonzero(sample["scribble"]) == 1
    legacy_condition = "m0" + "_only"
    with pytest.raises(LearningContractError, match="unknown editor condition"):
        EpisodeDataset(
            manifest,
            "test",
            editor_condition=legacy_condition,
            load_evaluation=False,
        )


def test_visible_npz_rejects_embedded_gt(tmp_path: Path) -> None:
    visible = tmp_path / "visible.npz"
    np.savez_compressed(visible, visual=np.zeros((17, 8, 8), np.float32), m0=np.zeros((8, 8)), scribble=np.zeros((8, 8)), cue_fg=np.zeros((8, 8)), cue_bg=np.zeros((8, 8)), spacing_xy=np.ones(2), gt=np.zeros((8, 8)))
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        '{"episode_id":"ep","patient_id":"p","partition":"test","goal":"ADD_SAME_LOCAL","operation":"ADD","target":"SAME","scope":"LOCAL","visible_npz":"%s","visible_sha256":"%s","evaluation_npz":"missing","evaluation_sha256":"missing"}\n' % (str(visible).replace("\\", "\\\\"), sha256_file(visible)),
        encoding="utf-8",
    )
    with pytest.raises(LearningContractError, match="schema mismatch"):
        EpisodeDataset(manifest, "test", load_evaluation=False)[0]


def test_visible_npz_rejects_nonfinite_and_nonbinary_arrays(tmp_path: Path) -> None:
    visible = tmp_path / "visible.npz"
    visual = np.zeros((17, 8, 8), np.float32)
    visual[0, 0, 0] = np.nan
    np.savez_compressed(
        visible,
        visual=visual,
        m0=np.zeros((8, 8), np.uint8),
        scribble=np.eye(8, dtype=np.uint8),
        cue_fg=np.eye(8, dtype=np.uint8),
        cue_bg=np.zeros((8, 8), dtype=np.uint8),
        spacing_xy=np.ones(2),
    )
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps({"episode_id":"ep","patient_id":"p","partition":"test","goal":"ADD_SAME_LOCAL","operation":"ADD","target":"SAME","scope":"LOCAL","visible_npz":str(visible),"visible_sha256":sha256_file(visible),"evaluation_npz":"missing","evaluation_sha256":"missing","geometry":{"output_spacing_xy":[1.0,1.0]}}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(LearningContractError, match="NaN or Inf"):
        EpisodeDataset(manifest, "test", load_evaluation=False)[0]


def _write_nifti(path: Path, array: np.ndarray) -> None:
    nib.save(nib.Nifti1Image(array, np.eye(4)), str(path))


def _materialization_fixture(tmp_path: Path):
    config = json.loads(
        (PROJECT / "configs" / "petct_route_a_experiment.json").read_text(
            encoding="utf-8"
        )
    )
    config_path = tmp_path / "experiment.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    shape = (9, 9, 3)
    arrays = {
        "ct": np.zeros(shape, dtype=np.float32),
        "pet": np.ones(shape, dtype=np.float32),
        "m0": np.zeros(shape, dtype=np.uint8),
        "gt": np.zeros(shape, dtype=np.uint8),
        "authorized": np.zeros(shape, dtype=np.uint8),
    }
    arrays["gt"][4, 4, 1] = 1
    arrays["authorized"][4, 4, 1] = 1
    paths = {}
    for name, array in arrays.items():
        path = tmp_path / (name + ".nii.gz")
        _write_nifti(path, array)
        paths[name] = path
    visible_document = tmp_path / "visible.json"
    evaluation_document = tmp_path / "evaluation.json"
    visible_document.write_text('{"visible":true}\n', encoding="utf-8")
    evaluation_document.write_text('{"evaluation":true}\n', encoding="utf-8")
    row = {
        "case_id": "case-a",
        "episode_id": "ep-a",
        "matched_state_group_id": "matched-state-a",
        "patient_id": "patient-a",
        "partition": "train",
        "held_out_fold": 0,
        "strategy": "centerline",
        "goal": "ADD_NEW_COMPLETE",
        "operation": "ADD",
        "target": "NEW",
        "scope": "COMPLETE",
        "coordinates_xyz": [[4, 4, 1]],
        "experiment_config_sha256": sha256_file(config_path),
        "visible_document": str(visible_document),
        "visible_document_sha256": sha256_file(visible_document),
        "evaluation_document": str(evaluation_document),
        "evaluation_document_sha256": sha256_file(evaluation_document),
        "scribble_generation": {
            "contract_version": "PETCT-SCRIBBLE-GENERATION-v2.0",
            "stage_order": list(tensor_materializer.__dict__.get("GENERATION_STAGE_ORDER", [])),
        },
    }
    for name, path in paths.items():
        row[name + "_path"] = str(path)
        row[name + "_sha256"] = sha256_file(path)
    learning_split = tmp_path / "learning-split.json"
    learning_split.write_text(
        json.dumps(
            {
                "schema_version": "PETCT-LEARNING-SPLIT-v1.0",
                "status": "FROZEN_BEFORE_MODEL_SELECTION",
                "patient_count": 1,
                "case_count": 1,
                "case_counts": {"train": 1, "val": 0, "test": 0},
                "patients": [
                    {
                        "patient_id": "patient-a",
                        "partition": "train",
                        "case_ids": ["case-a"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    row["learning_split_sha256"] = sha256_file(learning_split)
    manifest = tmp_path / "episodes.jsonl"
    outputs = {
        "visible": tmp_path / "learning-visible",
        "evaluation": tmp_path / "learning-evaluation",
        "manifest": tmp_path / "learning.jsonl",
    }
    argv = [
        "--episode-manifest",
        str(manifest),
        "--visible-root",
        str(outputs["visible"]),
        "--evaluation-root",
        str(outputs["evaluation"]),
        "--output-manifest",
        str(outputs["manifest"]),
        "--experiment-config",
        str(config_path),
        "--learning-split",
        str(learning_split),
        "--partitions",
        "train",
    ]
    return row, manifest, outputs, argv


def test_main_materializes_separated_outputs_and_publishes_manifest_last(
    tmp_path: Path, capsys
) -> None:
    row, source_manifest, outputs, argv = _materialization_fixture(tmp_path)
    source_manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
    assert main(argv) == 0
    assert json.loads(capsys.readouterr().out)["episodes"] == 1
    output_row = json.loads(outputs["manifest"].read_text(encoding="utf-8"))
    assert output_row["case_id"] == "case-a"
    materialized = json.loads(outputs["manifest"].read_text(encoding="utf-8"))
    assert Path(materialized["visible_npz"]).parent == outputs["visible"]
    assert Path(materialized["evaluation_npz"]).parent == outputs["evaluation"]
    assert Path(materialized["visible_npz"]).is_file()
    assert Path(materialized["evaluation_npz"]).is_file()
    assert materialized["scribble_generation"] == row["scribble_generation"]
    with np.load(materialized["visible_npz"], allow_pickle=False) as bundle:
        assert bundle["visual"].shape == (17, 256, 256)
        assert np.count_nonzero(bundle["visual"][10:15]) == 0
        assert np.array_equal(bundle["visual"][15], bundle["cue_fg"])
        assert np.count_nonzero(bundle["cue_bg"]) == 0


def test_main_materializes_natural_rows_without_matched_state_group(
    tmp_path: Path, capsys
) -> None:
    row, source_manifest, outputs, argv = _materialization_fixture(tmp_path)
    row.pop("matched_state_group_id")
    source_manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
    assert main(argv) == 0
    assert json.loads(capsys.readouterr().out)["episodes"] == 1
    materialized = json.loads(outputs["manifest"].read_text(encoding="utf-8"))
    assert "matched_state_group_id" not in materialized
    assert materialized["episode_id"] == "ep-a"


def test_main_rolls_back_staged_npz_when_later_episode_fails(
    tmp_path: Path,
) -> None:
    row, source_manifest, outputs, argv = _materialization_fixture(tmp_path)
    broken = dict(row)
    broken["episode_id"] = "ep-b"
    broken["ct_sha256"] = "0" * 64
    source_manifest.write_text(
        json.dumps(row) + "\n" + json.dumps(broken) + "\n", encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="source artifact hash mismatch"):
        main(argv)
    assert all(not path.exists() for path in outputs.values())
    assert not list(tmp_path.glob(".*.partial"))


@pytest.mark.parametrize(
    "partition,receipt_args",
    [
        ("test", []),
        ("val", ["--test-access-receipt", "not-allowed.json"]),
    ],
)
def test_partition_gate_precedes_episode_nifti_or_npz_read(
    tmp_path: Path, monkeypatch, partition: str, receipt_args: list[str]
) -> None:
    def forbidden(*args, **kwargs):
        raise AssertionError("data was read before partition authorization")

    monkeypatch.setattr(Path, "open", forbidden)
    monkeypatch.setattr(Path, "read_text", forbidden)
    monkeypatch.setattr(tensor_materializer, "load_jsonl", forbidden)
    monkeypatch.setattr(tensor_materializer.nib, "load", forbidden)
    monkeypatch.setattr(tensor_materializer.np, "load", forbidden)
    argv = [
        "--episode-manifest", str(tmp_path / "episodes.jsonl"),
        "--visible-root", str(tmp_path / "visible"),
        "--evaluation-root", str(tmp_path / "evaluation"),
        "--output-manifest", str(tmp_path / "tensors.jsonl"),
        "--experiment-config", str(tmp_path / "experiment.json"),
        "--learning-split", str(tmp_path / "split.json"),
        "--partitions", partition,
        *receipt_args,
    ]
    with pytest.raises(SystemExit):
        tensor_materializer.main(argv)


def test_verified_source_path_rejects_symlink_before_resolve(
    tmp_path: Path, monkeypatch
) -> None:
    link = tmp_path / "link.nii.gz"
    original_is_symlink = Path.is_symlink
    original_resolve = Path.resolve
    monkeypatch.setattr(
        Path, "is_symlink", lambda self: self == link or original_is_symlink(self)
    )

    def guarded_resolve(self, *args, **kwargs):
        if self == link:
            raise AssertionError("symlink was resolved before it was rejected")
        return original_resolve(self, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", guarded_resolve)
    with pytest.raises(RuntimeError, match="non-symlink"):
        tensor_materializer._verified_source_path(
            {"gt_path": str(link), "gt_sha256": "0" * 64},
            "gt_path",
            "gt_sha256",
        )
