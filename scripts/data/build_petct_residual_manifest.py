#!/usr/bin/env python3
"""Derive FN/FP residuals only from a validated patient-excluded OOF M0 gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence
from uuid import uuid4

import nibabel as nib
import numpy as np

SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))
for support_dir in (SCRIPTS_ROOT / "baseline", SCRIPTS_ROOT / "data"):
    if str(support_dir) not in sys.path:
        sys.path.insert(0, str(support_dir))

from baseline.validate_petct_m0_oof import (  # noqa: E402
    build_natural_oof_binding_from_validated,
    validate_oof_case_leaf,
    validate_oof_ready_receipt_only,
)
from common.petct_learning import load_jsonl, sha256_file  # noqa: E402
from common.petct_route_a_core import residual_masks, validate_patient_folds  # noqa: E402
from common.petct_test_access import (  # noqa: E402
    TestAccessError,
    add_leaf_test_access_arguments,
    enforce_partition_access,
)
from data.validate_petct_learning_split import (  # noqa: E402
    load_and_validate_learning_split,
)


READY_SCHEMA = "PETCT-FN-FP-RESIDUAL-READY-v2.0"
READY_PHASE = "FN_FP_RESIDUAL_DERIVATION"


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def _record(path: Path, *, display_path: Path | None = None) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"receipt-bound file is missing: {path}")
    return {
        "path": str((display_path or path).resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _cohort_bucket(case_ids: Sequence[str], source: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    cases = sorted(set(str(case_id) for case_id in case_ids))
    patients = sorted({str(source[case_id]["patient_id"]).casefold() for case_id in cases})
    return {
        "case_count": len(cases),
        "patient_count": len(patients),
        "case_ids": cases,
        "patient_ids": patients,
        "case_ids_sha256": _canonical_sha256(cases),
        "patient_ids_sha256": _canonical_sha256(patients),
    }


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
    """Stage a multi-path product and publish its manifest last.

    Every staging path is a unique sibling of its final destination, so each
    individual rename is same-filesystem and atomic.  A failed build removes
    all staging paths; a failed commit rolls back only paths owned by this run.
    """

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


def resolve_oof_mask(validated: Mapping[str, Any], case_record: Mapping[str, Any]) -> Path:
    """Resolve and rehash a mask record already accepted by OOF_READY."""

    run_dir = Path(str(validated["run_dir"])).resolve()
    mask_record = case_record.get("mask")
    raw = mask_record.get("path") if isinstance(mask_record, Mapping) else None
    if not isinstance(raw, str) or not raw:
        raise RuntimeError("OOF case mask record is missing")
    candidate = Path(raw)
    if not candidate.is_absolute() and ".." in candidate.parts:
        raise RuntimeError("unsafe relative OOF mask path")
    raw_path = candidate if candidate.is_absolute() else run_dir / candidate
    if raw_path.is_symlink():
        raise RuntimeError("OOF mask must be a regular non-symlink file")
    path = raw_path.resolve()
    if not path.is_relative_to(run_dir) or not path.is_file():
        raise RuntimeError("OOF mask escapes or is missing from its committed run")
    if path.stat().st_size != mask_record.get("bytes") or sha256_file(path) != mask_record.get("sha256"):
        raise RuntimeError("OOF mask changed after OOF_READY")
    return path


def _regular_source_path(raw: Any, *, label: str) -> Path:
    candidate = Path(str(raw or ""))
    if candidate.is_symlink():
        raise RuntimeError(f"{label} must be a regular non-symlink file")
    path = candidate.resolve()
    if not path.is_file():
        raise RuntimeError(f"{label} must be a regular non-symlink file: {path}")
    return path


def write_binary_nifti(path: Path, mask: np.ndarray, reference: nib.Nifti1Image) -> None:
    header = reference.header.copy()
    header.set_data_dtype(np.uint8)
    nib.save(
        nib.Nifti1Image(mask.astype(np.uint8), reference.affine, header),
        str(path),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oof-ready", type=Path, required=True)
    parser.add_argument("--learning-split", type=Path, required=True)
    parser.add_argument("--experiment-config", type=Path, required=True)
    parser.add_argument(
        "--case-manifest",
        type=Path,
        required=True,
        help="JSONL with case/patient/fold, partition, goal, CT/PET and GT paths",
    )
    parser.add_argument(
        "--partitions",
        nargs="+",
        choices=("train", "val", "test"),
        required=True,
        help="explicit partitions to materialize; omitted partitions are never opened",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument(
        "--ready-receipt",
        type=Path,
        required=True,
        help="no-clobber PETCT-FN-RESIDUAL-READY receipt published last",
    )
    add_leaf_test_access_arguments(parser)
    args = parser.parse_args(argv)
    if len(args.partitions) != len(set(args.partitions)):
        parser.error("--partitions must not contain duplicates")
    output_dir = args.output_dir.resolve()
    output_manifest = args.output_manifest.resolve()
    ready_receipt = args.ready_receipt.resolve()
    selected_partition_order = tuple(args.partitions)
    selected_partitions = set(selected_partition_order)
    # This authorization gate deliberately precedes OOF_READY, source-manifest,
    # split, NIfTI, hash, and output inspection.  A locked test request must not
    # learn even metadata from those artifacts before its consumed receipt is
    # revalidated.
    try:
        test_access = enforce_partition_access(
            selected_partitions,
            receipt_path=args.test_access_receipt,
            experiment_config=args.experiment_config,
            learning_split=args.learning_split,
            run_root=args.run_root,
            output_paths=(output_dir, output_manifest, ready_receipt),
        )
    except TestAccessError as error:
        parser.error(str(error))
    if (
        _path_lexists(output_dir)
        or _path_lexists(output_manifest)
        or _path_lexists(ready_receipt)
    ):
        parser.error("output path already exists")

    with args.experiment_config.open("r", encoding="utf-8") as stream:
        experiment_config = json.load(stream)
    source_rows = load_jsonl(args.case_manifest)
    _, learning_split = load_and_validate_learning_split(
        args.learning_split, source_rows, experiment_config
    )
    validated = validate_oof_ready_receipt_only(args.oof_ready)
    cohort = validate_patient_folds(source_rows)
    if cohort["case_count"] != 597 or cohort["patient_count"] != 378:
        raise RuntimeError("natural residual cohort must be exactly 597 cases / 378 patients")
    source = {str(row["case_id"]): row for row in source_rows}
    if len(source) != len(source_rows) or set(source) != set(validated["cases"]):
        raise RuntimeError("case manifest must match the complete OOF_READY inventory")
    experiment_config_sha256 = sha256_file(args.experiment_config)
    test_access_sha256 = (
        None if test_access is None else str(test_access["receipt_sha256"])
    )
    rows = []
    with staged_output_bundle(
        directory_outputs=[output_dir], file_outputs=[output_manifest, ready_receipt]
    ) as staged:
        staged_dir = staged[output_dir]
        staged_manifest = staged[output_manifest]
        staged_ready = staged[ready_receipt]
        for case_id in sorted(source):
            row = source[case_id]
            partition = learning_split["case_to_partition"][case_id]
            if "partition" in row and row["partition"] != partition:
                raise RuntimeError("case-manifest partition differs from frozen learning split")
            # This is deliberately before resolving, hashing, or opening GT.
            # A locked test case may exist in the cohort manifest but cannot be
            # materialized (or even have its truth read) by a development run.
            if partition not in selected_partitions:
                continue
            oof = validated["cases"][case_id]
            patient_id = str(row["patient_id"])
            if patient_id.casefold() != oof["patient_id"] or int(row["held_out_fold"]) != int(oof["held_out_fold"]):
                raise RuntimeError("case patient/fold differs from OOF_READY")
            truth_binding = validate_oof_case_leaf(
                validated,
                ready_path=args.oof_ready,
                case_id=case_id,
                source_record=row,
            )
            m0_path = Path(truth_binding["m0"]["path"])
            provenance = build_natural_oof_binding_from_validated(
                validated,
                ready_path=args.oof_ready,
                case_id=case_id,
                patient_id=patient_id,
                m0_path=m0_path,
                leaf_binding=truth_binding,
            )
            gt_path = Path(truth_binding["inputs"]["gt"]["path"])
            ct_sha256 = truth_binding["inputs"]["ct"]["sha256"]
            pet_sha256 = truth_binding["inputs"]["pet"]["sha256"]
            gt_image, m0_image = nib.load(str(gt_path)), nib.load(str(m0_path))
            if gt_image.shape != m0_image.shape or not np.allclose(
                gt_image.affine, m0_image.affine, atol=1e-3, rtol=0
            ):
                raise RuntimeError("GT/M0 geometry mismatch for %s" % case_id)
            residual = residual_masks(
                np.asarray(gt_image.dataobj) > 0,
                np.asarray(m0_image.dataobj) > 0,
            )
            fn_staged_path = staged_dir / (case_id + "_fn.nii.gz")
            fp_staged_path = staged_dir / (case_id + "_fp.nii.gz")
            fn_final_path = output_dir / fn_staged_path.name
            fp_final_path = output_dir / fp_staged_path.name
            write_binary_nifti(fn_staged_path, residual["fn"], gt_image)
            write_binary_nifti(fp_staged_path, residual["fp"], gt_image)
            rows.append(
                {
                    **{key: value for key, value in row.items() if key != "partition"},
                    "partition": partition,
                    "learning_split_sha256": learning_split["split_sha256"],
                    "learning_split_receipt": {
                        key: learning_split[key]
                        for key in (
                            "algorithm",
                            "seed",
                            "target_patient_counts",
                            "patient_counts",
                            "case_counts",
                        )
                    },
                    "experiment_config_sha256": experiment_config_sha256,
                    "test_access_receipt_sha256": (
                        test_access_sha256 if partition == "test" else None
                    ),
                    "m0_path": str(m0_path),
                    "m0_provenance": provenance,
                    "truth_binding": truth_binding,
                    "fn_path": str(fn_final_path),
                    "fn_sha256": sha256_file(fn_staged_path),
                    "fp_path": str(fp_final_path),
                    "fp_sha256": sha256_file(fp_staged_path),
                    "fn_voxels": int(residual["fn"].sum()),
                    "fp_voxels": int(residual["fp"].sum()),
                    "gt_sha256": truth_binding["inputs"]["gt"]["sha256"],
                    "ct_sha256": ct_sha256,
                    "pet_sha256": pet_sha256,
                    "m0_sha256": truth_binding["m0"]["sha256"],
                    "residual_contract": (
                        "FN=GT\\M0 and FP=M0\\GT are mining masks only; "
                        "episode-specific authorized target is bound after scribble selection"
                    ),
                }
            )

        with staged_manifest.open("x", encoding="utf-8", newline="\n") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

        all_case_ids = sorted(source)
        selected_case_ids = sorted(
            case_id
            for case_id in source
            if learning_split["case_to_partition"][case_id] in selected_partitions
        )
        generated_case_ids = sorted(str(row["case_id"]) for row in rows)
        fn_positive_case_ids = sorted(
            str(row["case_id"]) for row in rows if int(row["fn_voxels"]) > 0
        )
        zero_fn_case_ids = sorted(
            str(row["case_id"]) for row in rows if int(row["fn_voxels"]) == 0
        )
        fp_positive_case_ids = sorted(
            str(row["case_id"]) for row in rows if int(row["fp_voxels"]) > 0
        )
        zero_fp_case_ids = sorted(
            str(row["case_id"]) for row in rows if int(row["fp_voxels"]) == 0
        )
        excluded_case_ids = sorted(set(all_case_ids) - set(selected_case_ids))
        if generated_case_ids != selected_case_ids:
            raise RuntimeError("generated residual inventory differs from selected source cohort")
        if set(fn_positive_case_ids) | set(zero_fn_case_ids) != set(generated_case_ids):
            raise RuntimeError("FN-positive/zero-FN accounting does not cover generated residuals")
        if set(fp_positive_case_ids) | set(zero_fp_case_ids) != set(generated_case_ids):
            raise RuntimeError("FP-positive/zero-FP accounting does not cover generated residuals")
        source_bucket = _cohort_bucket(all_case_ids, source)
        selected_bucket = _cohort_bucket(selected_case_ids, source)
        generated_bucket = _cohort_bucket(generated_case_ids, source)
        fn_positive_bucket = _cohort_bucket(fn_positive_case_ids, source)
        zero_fn_bucket = _cohort_bucket(zero_fn_case_ids, source)
        fp_positive_bucket = _cohort_bucket(fp_positive_case_ids, source)
        zero_fp_bucket = _cohort_bucket(zero_fp_case_ids, source)
        excluded_bucket = _cohort_bucket(excluded_case_ids, source)
        excluded_bucket["reasons"] = {
            "PARTITION_NOT_SELECTED": {
                **_cohort_bucket(excluded_case_ids, source),
                "description": "case belongs to a frozen learning partition not requested by this run",
            }
        }
        ready = {
            "schema_version": READY_SCHEMA,
            "status": "PASS",
            "phase": READY_PHASE,
            "selected_partitions": list(selected_partition_order),
            "oof_ready": _record(args.oof_ready.resolve()),
            "source_case_manifest": _record(args.case_manifest.resolve()),
            "learning_split_sha256": learning_split["split_sha256"],
            "experiment_config_sha256": experiment_config_sha256,
            "test_access_receipt_sha256": test_access_sha256,
            "residual_directory": str(output_dir),
            "residual_manifest": _record(
                staged_manifest, display_path=output_manifest
            ),
            "cohort": {
                "source": source_bucket,
                "selected_source": selected_bucket,
                "generated": generated_bucket,
                "fn_positive": fn_positive_bucket,
                "zero_fn": zero_fn_bucket,
                "fp_positive": fp_positive_bucket,
                "zero_fp": zero_fp_bucket,
                "excluded": excluded_bucket,
            },
            "survivor_coverage": {
                "add_case_fraction": (
                    fn_positive_bucket["case_count"] / selected_bucket["case_count"]
                    if selected_bucket["case_count"]
                    else None
                ),
                "add_patient_fraction": (
                    fn_positive_bucket["patient_count"] / selected_bucket["patient_count"]
                    if selected_bucket["patient_count"]
                    else None
                ),
                "remove_case_fraction": (
                    fp_positive_bucket["case_count"] / selected_bucket["case_count"]
                    if selected_bucket["case_count"]
                    else None
                ),
                "remove_patient_fraction": (
                    fp_positive_bucket["patient_count"] / selected_bucket["patient_count"]
                    if selected_bucket["patient_count"]
                    else None
                ),
            },
        }
        with staged_ready.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(ready, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
    print(
        json.dumps(
            {
                "cases": len(rows),
                "patients": len({str(row["patient_id"]).casefold() for row in rows}),
                "fn_positive_cases": sum(row["fn_voxels"] > 0 for row in rows),
                "fp_positive_cases": sum(row["fp_voxels"] > 0 for row in rows),
                "ready_receipt": str(ready_receipt),
                "source": "validated patient-excluded OOF_READY",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
