#!/usr/bin/env python3
"""Run one actual-case Dataset901 inference/probability smoke prediction."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Callable, Sequence


FOLD = 0
CHECKPOINT_NAME = "checkpoint_final.pth"
EXPECTED_TRAINER = "nnUNetTrainer_1epoch"


def _require_regular_nonempty(path: Path, label: str) -> Path:
    path = path.resolve()
    if path.is_symlink() or not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"{label} must be a non-empty regular file: {path}")
    return path


def run_inference_smoke(
    *,
    model_training_output_dir: Path,
    case_id: str,
    image_files: Sequence[Path],
    output_dir: Path,
    predictor_factory: Callable[..., Any] | None = None,
    torch_module: Any | None = None,
) -> dict[str, Any]:
    """Use nnU-Net's official raw-data predictor for exactly one fold-0 case."""

    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,255}", case_id):
        raise ValueError("unsafe inference-smoke case identifier")
    if len(image_files) != 2:
        raise ValueError("inference smoke requires exactly CT and PET image files")
    resolved_images = [
        _require_regular_nonempty(Path(path), f"input channel {index}")
        for index, path in enumerate(image_files)
    ]
    model_training_output_dir = model_training_output_dir.resolve()
    if model_training_output_dir.is_symlink() or not model_training_output_dir.is_dir():
        raise RuntimeError("model training output directory is missing")
    output_dir = output_dir.resolve()
    if output_dir.is_symlink() or not output_dir.is_dir():
        raise RuntimeError("prediction output directory must already exist")
    if any(output_dir.iterdir()):
        raise RuntimeError("prediction output directory must be empty")
    if os.environ.get("nnUNet_compile") != "false":
        raise RuntimeError(
            "inference smoke requires explicit nnUNet_compile=false; "
            "compiled inference is outside this frozen gate"
        )

    if torch_module is None:
        import torch as torch_module  # type: ignore[no-redef]
    if not torch_module.cuda.is_available():
        raise RuntimeError("CUDA is required for actual-case inference smoke")
    if predictor_factory is None:
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

        predictor_factory = nnUNetPredictor

    device = torch_module.device("cuda")
    predictor = predictor_factory(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=True,
        device=device,
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=False,
    )
    predictor.initialize_from_trained_model_folder(
        str(model_training_output_dir),
        use_folds=(FOLD,),
        checkpoint_name=CHECKPOINT_NAME,
    )
    trainer_name = getattr(predictor, "trainer_name", None)
    if trainer_name is not None and trainer_name != EXPECTED_TRAINER:
        raise RuntimeError(
            "inference smoke resolved an unexpected trainer: " f"{trainer_name!r}"
        )

    output_prefix = output_dir / case_id
    input_payload = [[str(path) for path in resolved_images]]
    output_payload = [str(output_prefix)]
    predictor.predict_from_files(
        input_payload,
        output_payload,
        save_probabilities=True,
        overwrite=False,
        num_processes_preprocessing=2,
        num_processes_segmentation_export=2,
        folder_with_segs_from_prev_stage=None,
        num_parts=1,
        part_id=0,
    )

    outputs = {
        "binary_mask": output_prefix.with_suffix(".nii.gz"),
        "probabilities": output_prefix.with_suffix(".npz"),
        "properties": output_prefix.with_suffix(".pkl"),
    }
    for label, path in outputs.items():
        _require_regular_nonempty(path, label)

    return {
        "status": "PREDICTION_COMPLETED",
        "case_id": case_id,
        "fold": FOLD,
        "checkpoint_name": CHECKPOINT_NAME,
        "trainer": EXPECTED_TRAINER,
        "input_files": [str(path) for path in resolved_images],
        "output_prefix": str(output_prefix),
        "probability_exported": True,
        "compile_mode": "disabled",
        "nnunet_compile": "false",
        "scientific_metrics_computed": False,
        "oof_prediction_count": 0,
        "result_count": 0,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--ct", type=Path, required=True)
    parser.add_argument("--pet", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    receipt = run_inference_smoke(
        model_training_output_dir=args.model_dir,
        case_id=args.case_id,
        image_files=[args.ct, args.pet],
        output_dir=args.output_dir,
    )
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
