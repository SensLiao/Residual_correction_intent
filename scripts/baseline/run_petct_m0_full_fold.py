#!/usr/bin/env python3
"""Run one standard 1000-epoch Dataset901 nnU-Net fold."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable


DATASET_ID = 901
CONFIGURATION = "3d_fullres"
TRAINER = "nnUNetTrainer"
PLANS_IDENTIFIER = "nnUNetPlans"
NUM_EPOCHS = 1000
COMPILE_MODES = {"triton-stub-link", "disabled"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_stub(path: Path, expected_sha256: str) -> bool:
    return path.is_file() and _sha256(path) == expected_sha256


def _configure_compile_environment(
    *,
    mode: str,
    cuda_stub_dir: Path,
    expected_stub_sha256: str,
    environment: dict[str, str],
    stub_validator: Callable[[Path, str], bool],
) -> dict[str, Any]:
    if mode not in COMPILE_MODES:
        raise RuntimeError(f"unsupported compile mode: {mode}")
    stub_dir = cuda_stub_dir.resolve()
    stub = stub_dir / "libcuda.so"
    if mode == "disabled":
        environment["nnUNet_compile"] = "false"
        return {
            "mode": mode,
            "nnunet_compile": "false",
            "library_path_injection": False,
            "ld_library_path_stub_forbidden": True,
            "cuda_stub_dir": str(stub_dir),
            "cuda_stub_libcuda_path": None,
            "cuda_stub_libcuda_sha256": None,
        }

    if not expected_stub_sha256 or not stub_validator(stub, expected_stub_sha256):
        raise RuntimeError("compile-time libcuda stub is missing or changed")
    ld_entries = [
        Path(entry).resolve()
        for entry in environment.get("LD_LIBRARY_PATH", "").split(os.pathsep)
        if entry
    ]
    if stub_dir in ld_entries:
        raise RuntimeError("CUDA stub must never be placed in LD_LIBRARY_PATH")
    library_entries = [
        entry
        for entry in environment.get("LIBRARY_PATH", "").split(os.pathsep)
        if entry
    ]
    if str(stub_dir) not in library_entries:
        library_entries.insert(0, str(stub_dir))
    environment["LIBRARY_PATH"] = os.pathsep.join(library_entries)
    environment["nnUNet_compile"] = "true"
    return {
        "mode": mode,
        "nnunet_compile": "true",
        "library_path_injection": True,
        "ld_library_path_stub_forbidden": True,
        "cuda_stub_dir": str(stub_dir),
        "cuda_stub_libcuda_path": str(stub.resolve()),
        "cuda_stub_libcuda_sha256": expected_stub_sha256,
    }


def run_standard_fold(
    *,
    fold: int,
    resume: bool,
    actual_validation: bool,
    export_probabilities: bool,
    compile_mode: str,
    cuda_stub_dir: Path,
    expected_stub_sha256: str,
    environment: dict[str, str] | None = None,
    stub_validator: Callable[[Path, str], bool] = _validate_stub,
    trainer_factory: Callable[..., Any] | None = None,
    checkpoint_loader: Callable[..., None] | None = None,
    torch_module: Any | None = None,
) -> dict[str, Any]:
    """Run one official standard trainer fold without changing its defaults."""

    if fold not in range(5):
        raise ValueError("fold must be one of 0,1,2,3,4")
    if export_probabilities and not actual_validation:
        raise ValueError("probability export requires actual validation")
    if environment is None:
        environment = os.environ
    compile_contract = _configure_compile_environment(
        mode=compile_mode,
        cuda_stub_dir=cuda_stub_dir,
        expected_stub_sha256=expected_stub_sha256,
        environment=environment,
        stub_validator=stub_validator,
    )
    if torch_module is None:
        import torch as torch_module  # type: ignore[no-redef]
    if trainer_factory is None or checkpoint_loader is None:
        from nnunetv2.run.run_training import (
            get_trainer_from_args,
            maybe_load_checkpoint,
        )

        trainer_factory = trainer_factory or get_trainer_from_args
        checkpoint_loader = checkpoint_loader or maybe_load_checkpoint

    if not torch_module.cuda.is_available():
        raise RuntimeError("CUDA is required for standard fold training")
    torch_module.set_num_threads(1)
    torch_module.set_num_interop_threads(1)
    device = torch_module.device("cuda")
    trainer = trainer_factory(
        # nnU-Net v2.8.1 expects the CLI-style string before converting to int.
        dataset_name_or_id=str(DATASET_ID),
        configuration=CONFIGURATION,
        fold=fold,
        trainer_name=TRAINER,
        plans_identifier=PLANS_IDENTIFIER,
        continue_training=resume,
        device=device,
    )
    if trainer.num_epochs != NUM_EPOCHS:
        raise RuntimeError("standard nnUNetTrainer no longer resolves to 1000 epochs")
    if trainer.disable_checkpointing:
        raise RuntimeError("checkpointing must remain enabled for full training")
    if not trainer.output_folder:
        raise RuntimeError("official trainer did not resolve an output folder")

    checkpoint_loader(trainer, resume, False, None)
    torch_module.backends.cudnn.deterministic = False
    torch_module.backends.cudnn.benchmark = True
    trainer.run_training()
    if actual_validation:
        trainer.perform_actual_validation(export_probabilities)
    return {
        "status": "FOLD_PROCESS_COMPLETED",
        "dataset_id": DATASET_ID,
        "configuration": CONFIGURATION,
        "fold": fold,
        "trainer": TRAINER,
        "plans_identifier": PLANS_IDENTIFIER,
        "num_epochs": NUM_EPOCHS,
        "resume": resume,
        "actual_validation": actual_validation,
        "export_probabilities": export_probabilities,
        "compile_contract": compile_contract,
        "output_folder": str(trainer.output_folder),
    }


def _bool(value: str) -> bool:
    if value not in {"true", "false"}:
        raise argparse.ArgumentTypeError("expected true or false")
    return value == "true"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, choices=range(5), required=True)
    parser.add_argument("--resume", type=_bool, required=True)
    parser.add_argument("--actual-validation", type=_bool, required=True)
    parser.add_argument("--export-probabilities", type=_bool, required=True)
    parser.add_argument("--compile-mode", choices=sorted(COMPILE_MODES), required=True)
    parser.add_argument("--cuda-stub-dir", type=Path, required=True)
    parser.add_argument("--cuda-stub-sha256", default="")
    parser.add_argument("--runtime-receipt", type=Path, required=True)
    args = parser.parse_args()
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "TORCHINDUCTOR_COMPILE_THREADS",
    ):
        os.environ[name] = "1"
    payload = run_standard_fold(
        fold=args.fold,
        resume=args.resume,
        actual_validation=args.actual_validation,
        export_probabilities=args.export_probabilities,
        compile_mode=args.compile_mode,
        cuda_stub_dir=args.cuda_stub_dir,
        expected_stub_sha256=args.cuda_stub_sha256,
    )
    args.runtime_receipt.parent.mkdir(parents=True, exist_ok=True)
    with args.runtime_receipt.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
