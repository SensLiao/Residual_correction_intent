# Training SIRB

```
python training/train.py --config sirb/configs/train/sirb_psma.json
```

trains SIRB for 50,000 training iterations on the training set with the paper's recipe, with
ground-truth-corrected states in place of the auxiliary-model states (see [Data](#data)). The GPU is chosen with `CUDA_VISIBLE_DEVICES`; `--device cpu` runs on the
CPU (useful only to check a setup).

| File | Contents |
|---|---|
| `train.py` | entry point: `--config`, `--resume` or `--resume-from CHECKPOINT`, `--device`, then `key=value` overrides |
| `trainer.py` | the training loop: unit preparation in worker processes, training iterations, checkpoints, resuming, logs, self-generated states |
| `model/sirb_train.py` | `SIRBTrain`: SIRB with the forward pass and supervision of one view |
| `loss_fns.py` | native CE, Dice loss, source CE and the view total; the source targets |
| `optimizer.py` | AdamW and the learning-rate schedule |
| `state_refresh.py` | self-generated states from a frozen copy of the current model |

## One training iteration

Each training iteration trains on two units of four views: a state-pair unit (the two states of a stored state pair,
the same scribble on two masks; in one of every eight iterations a target-exhausted unit, whose second state has the
whole target error corrected) and a scribble-pair unit (one state with its own scribble and a second scribble that the
simulated user draws on another error; it falls back to the state pair when no second scribble works). A view is one
state in one 112-cell window of the 3 mm grid (`sirb/data/pairs.py`).

For every view in turn, SIRB builds the state context (the two windows of a state share it), runs the window, refines
the role logits of the editable native voxels inside the window (native refinement and the native head) and computes
the view total

    native CE + source CE + Dice loss

* native CE: cross-entropy of the native role logits over the window's editable valid native voxels, averaged over the
  roles present (target error, other errors, correct tissue); correct tissue is the mean of two strata, the correct
  tissue next to the target error and the rest;
* Dice loss on the target probability, when the window holds target-error voxels;
* source CE: cross-entropy of source-conditioned role prediction at 12 mm and 6 mm, each source supervised with the
  error components its own scribble voxels touch, from a bounded sample of the window's voxels.

Each view's loss is divided by the eight views and back-propagated before the next view; the gradients are then
clipped to a norm of 5 and AdamW takes one step. The learning rate rises linearly to 2e-4 over the first 1,000
iterations, then follows a cosine to 5e-5 at 40,000 and a second cosine to 2e-5 at 50,000.

## Data

Training reads, under `SIRB_PREPROCESSED`:

1. the examination cache, from `python tools/prepare_data.py --split train` (and `--split val` for evaluation);
2. the ground-truth-corrected states, from `python tools/generate_states.py --split train`;
3. the self-generated states, which the trainer makes itself (below).

The paper's recipe also drew on **auxiliary-model states** in the first 25,000 iterations: the states of five-round
interactions of auxiliary SIRB models trained on the training set, which supply model-made errors before the model can
make useful ones of its own. The released configuration gives their slots to the ground-truth-corrected states
(`data.mixing.earlier_version_pools: []`). Pools of states in the same format, from any model or error generator, can be
added with `data.mixing.earlier_version_pools='["earlier_version","earlier_version_gap_weighted"]'` (their folders are
`data.state_sources.earlier_version*`).

Which pool a unit draws from: in each block of eight iterations, every unit kind gets eight slots in a random order,
four for the ground-truth-corrected states, two for each patient half's newest self-generated states once they exist,
and the rest for the ground-truth-corrected states (or the auxiliary-model states, when given) (`data.mixing`).

### Self-generated states

After training iterations 10,000, 25,000 and 40,000 (`data.self_generated.refresh`) the trainer saves a checkpoint and
runs a frozen copy of the current model on every training examination of one patient half (half 0, then 1, then 0):
five correction rounds with the simulated user in the examination's drawing style, the predictor options of
`train.refresh.predictor` (every window of the examination, flip averaging, FP32: the computation of the paper, set
there in full and not taken from the inference modes). The states of these rounds and their state pairs go to
`SIRB_PREPROCESSED/states/self_generated/<run>/self_generated_half<h>_iteration<i>/`; once every examination of
the half is done the pool is published (`pool.json`) and replaces that half's earlier self-generated states in the
sampling from the next block of eight iterations on. A refresh that stops part-way resumes with the examinations
that are not yet done.

## Configuration

`sirb/configs/train/sirb_psma.json` inherits the model (`sirb/configs/model/sirb.json`) and the data
(`sirb/configs/data/psma.json`) configurations. Its own keys:

| Key | Value | Meaning |
|---|---|---|
| `run` | `sirb_psma` | experiment folder name under `SIRB_RESULTS` |
| `train.seed` | 3407 | seed of the random streams that choose units, windows and samples |
| `train.iterations` | 50000 | training iterations |
| `train.views_per_iteration` | 8 | two units of four views |
| `train.tf32` | true | TF32 matrix products and convolutions on the GPU (FP32 tensors) |
| `train.optimizer` | lr 2e-4, betas (0.9, 0.999), eps 1e-8, weight decay 1e-5 | AdamW |
| `train.learning_rate` | warm-up 1,000; cosine segments to 5e-5 at 40,000 and 2e-5 at 50,000 | schedule |
| `train.gradient_clip_norm` | 5 | gradient norm clipping |
| `train.loss` | 1, 1, 1; Dice epsilon 1e-6 | weights of native CE, source CE and Dice loss |
| `train.checkpoint_every` | 1000 | checkpoint interval (iterations) |
| `train.log_every` | 20 | log interval (iterations) |
| `train.refresh` | simulator seed 42, draw on examinations without lesion, predictor options | self-generated states |
| `data.self_generated.refresh` | 10,000 (half 0), 25,000 (half 1), 40,000 (half 0) | when self-generated states are made |

`train.runtime` holds the machine settings; they change memory use and time, not the recipe, and may differ when a run
is resumed:

| Key | Default | Meaning |
|---|---|---|
| `workers` | 4 | processes that prepare units (0: in the training process) |
| `prefetch_depth` | 3 | iterations whose units are prepared ahead |
| `cpu_threads` | 2 | CPU threads of the training process |
| `cudnn_benchmark` | true | let cuDNN choose its fastest convolution algorithms |
| `voxel_chunk` | 2097152 | native voxels refined together in a view |
| `recompute_convolutions` | false | recompute the encoders and the decoder in the backward pass (less memory) |
| `recompute_native_voxels` | false | recompute native refinement chunk by chunk in the backward pass (less memory) |
| `source_chunk` | 4 | sources per chunk of the 6 mm role head |
| `recompute_role_head` | true | recompute the 6 mm role head chunks in the backward pass |

If the GPU runs out of memory, set `train.runtime.recompute_convolutions=true` and
`train.runtime.recompute_native_voxels=true`, or lower `train.runtime.voxel_chunk`.

Any key can be overridden after the options, for example

```
python training/train.py --config sirb/configs/train/sirb_psma.json run=sirb_psma_seed1 train.seed=1 \
    train.runtime.workers=8
```

## Outputs

`SIRB_RESULTS/<run>/`:

| Path | Contents |
|---|---|
| `config.json` | the merged configuration |
| `checkpoints/latest.pt` | the newest checkpoint: model, optimizer, random streams and sampling course, self-generated pools, timings |
| `checkpoints/iteration_NNNNNN.pt` | the checkpoints at each refresh and at the end; `iteration_050000.pt` is the trained model |
| `logs/train.log` | the log |
| `logs/train.jsonl` | one row per logged iteration: learning rate, pools, view losses and loss terms, gradient norm, timings |
| `complete.json` | written when the run is complete |

A checkpoint loads into a model with `sirb.build_sirb.build_sirb(config, "checkpoints/iteration_050000.pt")` and is
evaluated with `tools/evaluate.py --checkpoint ...`.

## Resuming

SIGINT (`Ctrl+C`) or SIGTERM finishes the current training iteration, saves `checkpoints/latest.pt` and stops.

```
python training/train.py --config sirb/configs/train/sirb_psma.json --resume
python training/train.py --config sirb/configs/train/sirb_psma.json --resume-from path/to/checkpoint.pt
```

A resumed run continues as if it had not stopped: the checkpoint holds the random streams and the sampling course as
they were right after its iteration was reserved (units are reserved ahead for preparation), so the following
iterations train on the same units. Only `paths` and `train.runtime` may differ from the configuration the
checkpoint was trained with. A new run refuses an experiment folder that already holds checkpoints.

## Variants

The four variants of the paper are switches of the same network and training (`sirb/configs/model/variants/`):

| Variant | Override | What changes in training |
|---|---|---|
| No scribble conditioning | `model.scribble_conditioning=false` | the network reads no scribble or interaction input; each view's window is centred on a valid native voxel drawn independently of the scribble; no source CE |
| No correct-tissue role | `model.correct_tissue_role=false` | the correct-tissue logit is fixed; correct-tissue voxels are not supervised |
| Merged non-target roles | `model.merged_non_target_roles=true` | other errors and correct tissue share one likelihood in the native CE and the source CE |
| Coarse roles only | `model.native_refinement=false` | native voxels take the coarse role logits; no native refinement |

Each variant is trained from scratch with the same recipe, in its own experiment folder, for example

```
python training/train.py --config sirb/configs/train/sirb_psma.json run=sirb_psma_no_correct_tissue_role \
    model.correct_tissue_role=false
```

The variant's self-generated states come from the variant itself.

## Reproducibility

With the same configuration and data a run trains on the same units, windows and samples, whatever the number of
workers and the prefetch depth: units are reserved in iteration order in the training process. On a GPU the
arithmetic itself is not bit-for-bit deterministic (`cudnn_benchmark`, atomic additions), so two runs agree to float
rounding at first and then drift apart. The released model was trained with the auxiliary-model states described
above; the released configuration uses ground-truth-corrected states in their place.
