#!/usr/bin/env python3
"""Run the bounded Dataset901 fold-0 nnU-Net one-epoch smoke.

The pinned nnU-Net command-line entry point performs full-case validation after
training.  This smoke deliberately calls the same official trainer factory and
trainer loop without that final prediction phase.  Patch-level validation loss,
checkpoints, and logs are retained; OOF predictions are not produced.
"""

from __future__ import annotations

import json
import os
from typing import Any, Callable


DATASET_ID = 901
CONFIGURATION = "3d_fullres"
FOLD = 0
TRAINER = "nnUNetTrainer_1epoch"
PLANS_IDENTIFIER = "nnUNetPlans"


def run_one_epoch_smoke(
    *,
    trainer_factory: Callable[..., Any] | None = None,
    torch_module: Any | None = None,
) -> dict[str, Any]:
    """Run exactly one official trainer epoch and skip actual-case validation."""

    if torch_module is None:
        import torch as torch_module  # type: ignore[no-redef]
    if trainer_factory is None:
        from nnunetv2.run.run_training import get_trainer_from_args

        trainer_factory = get_trainer_from_args

    if not torch_module.cuda.is_available():
        raise RuntimeError("CUDA is required for the fold-0 one-epoch smoke")
    torch_module.set_num_threads(1)
    torch_module.set_num_interop_threads(1)
    device = torch_module.device("cuda")
    trainer = trainer_factory(
        # nnU-Net v2.8.1 calls .startswith() before converting the ID to int,
        # so the official factory requires the CLI-style string here.
        dataset_name_or_id=str(DATASET_ID),
        configuration=CONFIGURATION,
        fold=FOLD,
        trainer_name=TRAINER,
        plans_identifier=PLANS_IDENTIFIER,
        continue_training=False,
        device=device,
    )
    if trainer.num_epochs != 1:
        raise RuntimeError("resolved trainer is not constrained to exactly one epoch")
    if trainer.disable_checkpointing:
        raise RuntimeError("checkpointing must remain enabled for the smoke gate")
    if not trainer.output_folder:
        raise RuntimeError("official trainer did not resolve a run-scoped output folder")

    trainer.run_training()
    return {
        "status": "TRAINING_LOOP_COMPLETED",
        "dataset_id": DATASET_ID,
        "configuration": CONFIGURATION,
        "fold": FOLD,
        "trainer": TRAINER,
        "plans_identifier": PLANS_IDENTIFIER,
        "epochs": 1,
        "output_folder": str(trainer.output_folder),
        "actual_validation": False,
        "export_probabilities": False,
    }


def main() -> int:
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "TORCHINDUCTOR_COMPILE_THREADS",
    ):
        os.environ[name] = "1"
    result = run_one_epoch_smoke()
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
