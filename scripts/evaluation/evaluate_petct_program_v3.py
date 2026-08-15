#!/usr/bin/env python3
"""Evaluate v3 compiler predictions and editor corrections (SCEP metrics).

Every metric row carries an explicit ``denominator_domain``:
  * ``2d_prompted_plane`` — metrics on the prompted axial slice (2D in-plane
    voxel denominators);
  * ``3d_full_volume``  — full-volume metrics (3D voxel denominators,
    requires --volume-root and --candidates for the selected component);
  * ``protocol_constant`` — ratios such as single_slice_ceiling that are
    protocol constants, never oracle Dice ceilings.

2D and 3D denominators are never mixed into one number.

Inputs:
  * --predictions: compiler inference artifact (jsonl), one legal call (or
    ABSTAIN) per episode_id;
  * --episodes: episodes manifest with partition/goal/operation/
    matched_state_group_id/visible_npz/evaluation_npz references;
  * --editor-predictions: jsonl mapping episode_id -> delta npz path (2D
    prompted-slice delta) and optionally -> volume delta path;
  * --candidates: visible component-candidate sidecar directory (for the
    selected-component 2D mask and 3D component joins);
  * --volume-root: controlled-state root for 3D joins
    (<volume-root>/<group>/<goal>/{m0,authorized,gt}.nii.gz, as materialized
    by the builder).  Without it, 3D metrics are reported absent.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from common.petct_program_contract import (  # noqa: E402
    NEW_CUE_SENTINEL,
    family_ids,
    goal_to_family_id,
)


def _load_jsonl(path: Path):
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_nifti_binary(path: Path) -> np.ndarray:
    import nibabel as nib

    array = np.asarray(nib.load(str(path)).dataobj)
    if array.ndim != 3:
        raise ValueError("mask must be 3D: %s" % path)
    return (array > 0).astype(np.uint8)


def _dice(binary_a: np.ndarray, binary_b: np.ndarray) -> float:
    intersection = float(np.logical_and(binary_a, binary_b).sum())
    total = float(binary_a.sum()) + float(binary_b.sum())
    return 2.0 * intersection / total if total else float("nan")


def _patient_balanced_macro_f1(true_ids, pred_ids, patients, classes):
    per_patient = {}
    for true_id, pred_id, patient in zip(true_ids, pred_ids, patients):
        slot = per_patient.setdefault(str(patient), {"tp": {}, "fp": {}, "fn": {}})
        if true_id == pred_id:
            slot["tp"][true_id] = slot["tp"].get(true_id, 0) + 1
        else:
            slot["fp"][pred_id] = slot["fp"].get(pred_id, 0) + 1
            slot["fn"][true_id] = slot["fn"].get(true_id, 0) + 1
    per_class = []
    for class_id in classes:
        f1s = []
        for slot in per_patient.values():
            tp = slot["tp"].get(class_id, 0)
            fp = slot["fp"].get(class_id, 0)
            fn = slot["fn"].get(class_id, 0)
            denom = 2 * tp + fp + fn
            f1s.append(2 * tp / denom if denom else 1.0)
        per_class.append(float(np.mean(f1s)))
    return float(np.mean(per_class))


def _compiler_metrics(predictions, rows):
    pred_by_episode = {str(p["episode_id"]): p for p in predictions}
    val_rows = [row for row in rows if row.get("partition") == "val"]
    true_ids, pred_ids, patients = [], [], []
    legal_count = 0
    pointer_recalls = []
    pointer_rows = 0
    for row in val_rows:
        episode_id = str(row["episode_id"])
        prediction = pred_by_episode.get(episode_id)
        if prediction is None:
            continue
        operation = str(row["operation"])
        gold_family = goal_to_family_id(str(row["goal"]), operation)
        family = str(prediction.get("family") or "")
        legal_families = family_ids(operation, include_repair=True)
        if family in legal_families:
            legal_count += 1
            pred_id = legal_families.index(family)
        else:
            pred_id = -1
        true_ids.append(gold_family)
        pred_ids.append(pred_id)
        patients.append(str(row.get("patient_id") or row.get("source_patient_id")))
        if operation == "ADD" and str(row["goal"]) != "ADD_NEW_COMPLETE":
            pointer_rows += 1
            target_sets = row.get("pointer_targets") or []
            predicted_index = prediction.get("pointer_index")
            if target_sets and predicted_index is not None:
                pointer_recalls.append(
                    1.0 if int(predicted_index) in [int(t) for t in target_sets] else 0.0
                )
    covered = [index for index, value in enumerate(pred_ids) if value >= 0]
    macro_f1 = (
        _patient_balanced_macro_f1(
            [true_ids[i] for i in covered],
            [pred_ids[i] for i in covered],
            [patients[i] for i in covered],
            [0, 1, 2],
        )
        if covered
        else float("nan")
    )
    pairs = 0
    consistent = 0
    by_group = {}
    for row in val_rows:
        episode_id = str(row["episode_id"])
        if episode_id not in pred_by_episode:
            continue
        by_group.setdefault(str(row.get("matched_state_group_id") or ""), []).append(
            (episode_id, row)
        )
    for group_rows in by_group.values():
        for a in range(len(group_rows)):
            for b in range(a + 1, len(group_rows)):
                episode_a, row_a = group_rows[a]
                episode_b, row_b = group_rows[b]
                if row_a["operation"] != row_b["operation"]:
                    continue
                family_a = str(pred_by_episode[episode_a].get("family") or "")
                family_b = str(pred_by_episode[episode_b].get("family") or "")
                legal = family_ids(str(row_a["operation"]), include_repair=True)
                if family_a not in legal or family_b not in legal:
                    continue
                pairs += 1
                gold_a = goal_to_family_id(str(row_a["goal"]), str(row_a["operation"]))
                gold_b = goal_to_family_id(str(row_b["goal"]), str(row_b["operation"]))
                if gold_a != gold_b and family_a != family_b:
                    consistent += 1
    return {
        "patient_balanced_family_macro_f1": macro_f1,
        "legal_call_rate": legal_count / max(1, len(true_ids)),
        "matched_pair_count": pairs,
        "matched_pair_consistency": (consistent / pairs) if pairs else float("nan"),
        "pointer_recall_at_1": (
            float(np.mean(pointer_recalls)) if pointer_recalls else float("nan")
        ),
        "pointer_episodes": pointer_rows,
        "val_episodes_covered": len(covered),
        "val_episodes_total": len(true_ids),
    }


def _selected_2d_mask(candidates_dir: Path, episode_id: str, pointer_index):
    candidate_path = candidates_dir / ("%s.json" % episode_id)
    if not candidate_path.is_file():
        raise FileNotFoundError("missing candidates record: %s" % candidate_path)
    with candidate_path.open(encoding="utf-8") as stream:
        record = json.load(stream)
    if pointer_index is None:
        raise ValueError("pointer_index required for selected component")
    component = record["components"][int(pointer_index)]
    return np.asarray(component["prompted_slice_mask"], dtype=np.uint8)


def _editor_metrics_2d(row, delta):
    """2D prompted-plane editor metrics with 2D denominators."""

    with np.load(row["evaluation_npz"], allow_pickle=False) as bundle:
        target = np.asarray(bundle["target"], dtype=np.float32) > 0
        gt = np.asarray(bundle["gt"], dtype=np.float32) > 0
        authorized = np.asarray(bundle["authorized"], dtype=np.float32) > 0
    with np.load(row["visible_npz"], allow_pickle=False) as bundle:
        m0 = np.asarray(bundle["m0"], dtype=np.float32) > 0
    operation = str(row["operation"])
    delta = delta > 0
    if operation == "ADD":
        corrected = m0 | delta
    else:
        corrected = m0 & ~delta
    dice_before = _dice(m0, gt)
    dice_after = _dice(corrected, gt)
    recovery = float(np.logical_and(delta, authorized).sum()) / max(
        1.0, float(authorized.sum())
    )
    nonselected_harm = float(np.logical_and(delta, ~(authorized | m0)).sum()) / max(
        1.0, float(delta.sum())
    ) if delta.sum() else 0.0
    return {
        "denominator_domain": "2d_prompted_plane",
        "dice_before": dice_before,
        "dice_after": dice_after,
        "delta_dice": dice_after - dice_before,
        "target_recovery": recovery,
        "nonselected_harm_2d": nonselected_harm,
    }


def _editor_metrics_3d(row, volume_delta_path: str):
    """3D full-volume editor metrics with 3D denominators."""

    group = str(row["matched_state_group_id"])
    goal = str(row["goal"])
    volume_root = Path(str(row["_volume_root"]))
    m0 = _load_nifti_binary(volume_root / group / goal / "m0.nii.gz")
    authorized = _load_nifti_binary(volume_root / group / goal / "authorized.nii.gz")
    gt = _load_nifti_binary(volume_root / group / goal / "gt.nii.gz")
    delta = _load_nifti_binary(Path(volume_delta_path))
    operation = str(row["operation"])
    corrected = (m0 | delta) if operation == "ADD" else (m0 & ~delta)
    dice_before = _dice(m0, gt)
    dice_after = _dice(corrected, gt)
    recovery = float(np.logical_and(delta, authorized).sum()) / max(
        1.0, float(authorized.sum())
    )
    return {
        "denominator_domain": "3d_full_volume",
        "dice_before": dice_before,
        "dice_after": dice_after,
        "delta_dice": dice_after - dice_before,
        "target_recovery_3d": recovery,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--editor-predictions", type=Path, default=None)
    parser.add_argument("--candidates", type=Path, default=None)
    parser.add_argument("--volume-root", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    predictions = _load_jsonl(args.predictions)
    rows = _load_jsonl(args.episodes)
    compiler = _compiler_metrics(predictions, rows)
    editor_2d_rows = []
    editor_3d_rows = []
    if args.editor_predictions:
        editor_map = {
            str(entry["episode_id"]): entry for entry in _load_jsonl(args.editor_predictions)
        }
        for row in rows:
            if row.get("partition") != "val":
                continue
            episode_id = str(row["episode_id"])
            entry = editor_map.get(episode_id)
            if entry is None:
                continue
            delta_path = entry.get("delta_path")
            if delta_path:
                editor_2d_rows.append(
                    {"episode_id": episode_id, **_editor_metrics_2d(row, np.asarray(np.load(delta_path)["delta"]))}
                )
            if args.volume_root and entry.get("volume_delta_path"):
                row_copy = dict(row)
                row_copy["_volume_root"] = str(args.volume_root)
                editor_3d_rows.append(
                    {
                        "episode_id": episode_id,
                        **_editor_metrics_3d(row_copy, str(entry["volume_delta_path"])),
                    }
                )
    editor = {
        "2d_prompted_plane": editor_2d_rows,
        "3d_full_volume": editor_3d_rows,
        "3d_status": "reported" if editor_3d_rows else (
            "absent_without_volume_root" if not args.volume_root else "no_matching_predictions"
        ),
    }
    if editor_2d_rows:
        editor["2d_summary"] = {
            "mean_delta_dice_2d": float(
                np.nanmean([row["delta_dice"] for row in editor_2d_rows])
            ),
            "mean_target_recovery_2d": float(
                np.nanmean([row["target_recovery"] for row in editor_2d_rows])
            ),
            "episodes": len(editor_2d_rows),
        }
    if editor_3d_rows:
        editor["3d_summary"] = {
            "mean_delta_dice_3d": float(
                np.nanmean([row["delta_dice"] for row in editor_3d_rows])
            ),
            "episodes": len(editor_3d_rows),
        }
    report = {
        "schema_version": "PETCT-PROGRAM-EVAL-v1.0",
        "denominator_domains": {
            "2d_prompted_plane": "2D in-plane voxel denominators on the prompted axial slice",
            "3d_full_volume": "3D voxel denominators over the full volume (separate rows; never averaged with 2D)",
            "single_slice_ceiling": "protocol constant (authorized_operable_voxels / authorized_full_voxels); never an oracle Dice ceiling",
        },
        "compiler": compiler,
        "editor": editor,
        "predictions_sha256": _sha256_file(args.predictions),
        "episodes_sha256": _sha256_file(args.episodes),
    }
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
