# SIRB: Intent-Aware Interactive Correction of Whole-Body PSMA PET/CT Segmentations

**The University of Sydney**

Ruixuan Liao, Jinman Kim

[[`Dataset`](https://doi.org/10.57754/FDAT.g27kx-86t35)] [[`nnU-Net`](https://github.com/MIC-DKFZ/nnUNet)] [[`autoPET V`](https://www.autopet.org/autopetv.html)] [[`BibTeX`](#citing-sirb)]

![Interactive correction with SIRB](assets/workflow.png?raw=true)

**State-Interventional Referent Binding (SIRB)** corrects an existing 3D lesion segmentation of a whole-body PSMA
PET/CT examination from scribbles. In each round the user reviews the current mask and draws one scribble on an axial
slice: an addition scribble on a missed lesion, or a removal scribble on a false-positive region. SIRB reads the
scribble as a correction intent. It predicts three roles together (the target error that the user selected, other
errors and correct tissue), and its update rule applies the predicted target error directly as the correction. Each correction therefore follows the error the user chose and keeps what the current
mask already labels correctly.

![Correction intent and the three voxel roles](assets/intent_roles.png?raw=true)

## Highlights

- **Corrects the target error.** From the same nnU-Net initial segmentation, SIRB corrects 71.3% of the target error
  per scribble on average and introduces 3.27 mL of new error, against 59.3% and 6.43 mL for 2S-ICR.
- **Keeps the lesions already detected.** SIRB loses 0.7% of the detected lesions in at least one of five rounds,
  against 8.4% for 2S-ICR and 7.7% for UAM.
- **Highest Dice after ten rounds.** SIRB reaches a Dice of 0.844 after ten rounds, the highest of the seven methods
  in the table.
- **One model for any initial segmentation.** Without retraining, SIRB corrects 70.8% and 71.6% of the target error
  per scribble on the initial segmentations of 2S-ICR and UAM, against 57.9% and 57.0% for the two methods.
- **0.98 s per scribble** in the fast mode on one NVIDIA RTX 5090, faster than 2S-ICR, UAM and the scribble nnU-Nets,
  with one model each.

Scores are on the test set of PSMA-PET-CT-Lesions, 82 lesion-positive examinations from 56 patients, with the official autoPET V simulated user and averaged by patient. Times are measured on the validation
set.

## Latest updates

**11/2026 -- Code release**

- The SIRB model and its predictor with a standard and a fast mode, evaluation with the autoPET V simulated user over
  five or ten rounds, per-scribble timing, data preparation, the split script and the training code.

## Results

### Five rounds from the same initial segmentation

All methods start from the same nnU-Net initial segmentation and receive the same first scribble.

| **Method** | **Final Dice** ↑ | **nAUC-Dice** ↑ | **Target correction rate (%)** ↑ | **Edit precision (%)** ↑ | **New error (mL)** ↓ | **Net error reduction (mL)** ↑ |
| :--------- | :--------------: | :-------------: | :------------------------------: | :----------------------: | :------------------: | :----------------------------: |
| No correction | 0.64 | 0.64 | – | – | – | – |
| 2S-ICR | **0.81** | **0.76** | 59.3 | 62.8 | 6.43 | 1.55 |
| UAM | 0.79 | 0.73 | 57.4 | 61.6 | 6.05 | 1.86 |
| nnInteractive | 0.56 | 0.55 | 52.3 | 57.7 | 49.53 | −21.44 |
| SegAnyPET-Lesion | 0.49 | 0.52 | 43.5 | 43.6 | 23.15 | −7.49 |
| **SIRB (ours)** | **0.81** | **0.76** | **71.3** | **74.1** | **3.27** | **2.19** |

Target correction rate, edit precision, new error and net error reduction describe one scribble. The target
correction rate is the corrected share of the target error volume, edit precision the correct share of all changed
voxels, new error the volume of errors that did not exist before the scribble, and net error reduction the correct
change minus the new error. No correction keeps the initial segmentation unchanged and is not ranked.

![Volume changed per scribble](assets/changes_per_scribble.png?raw=true)

![First correction of each method](assets/comparison.png?raw=true)

*The first correction of each method from the same mask and scribble, for one addition (top) and one removal
(bottom). Green and hatched red show the errors left after the correction. On the whole-body projections, purple
marks correct change and orange marks new error.*

![Initially detected lesions lost after five rounds](assets/lesions_kept.png?raw=true)

*Lesions of the initial segmentation that each method loses after five rounds, in one examination with 139 detected
lesions.*

![A five-round SIRB interaction sequence](assets/five_round_case.png?raw=true)

### Ten rounds

Each method continues for ten rounds from its own initial segmentation. nnInteractive and SegAnyPET-Lesion start
from an empty mask and receive their own point prompts, and the own initial segmentation of the distance scribble
nnU-Net is almost empty, so their curves start at round 1.

| **Method** | **Dice after five rounds** ↑ | **Dice after ten rounds** ↑ |
| :--------- | :-----------------------: | :------------------------: |
| 2S-ICR | 0.810 | 0.834 |
| UAM | 0.790 | 0.816 |
| Binary scribble nnU-Net | 0.766 | 0.789 |
| Distance scribble nnU-Net | 0.749 | 0.765 |
| nnInteractive | 0.512 | 0.628 |
| SegAnyPET-Lesion | 0.355 | 0.384 |
| **SIRB (ours)** | **0.813** | **0.844** |

![Dice over ten correction rounds](assets/dice_over_ten_rounds.png?raw=true)

### Other initial segmentations

The same trained SIRB corrects the initial segmentations of 2S-ICR and UAM without retraining. Within each block
both methods start from the same mask and receive the same first scribble.

| **Initial segmentation** | **Method** | **Target correction rate (%)** ↑ | **New error (mL)** ↓ | **Detected lesions lost (%)** ↓ |
| :----------------------- | :--------- | :------------------------------: | :------------------: | :-----------------------------: |
| 2S-ICR | 2S-ICR | 57.9 | 5.24 | 7.5 |
| 2S-ICR | **SIRB (ours)** | **70.8** | **3.08** | **0.7** |
| UAM | UAM | 57.0 | 4.75 | 1.9 |
| UAM | **SIRB (ours)** | **71.6** | **3.48** | **0.8** |

### Time per scribble

![Time per scribble](assets/time_per_scribble.png?raw=true)

## Installation

SIRB needs `python>=3.10` and `torch>=2.4`. Install PyTorch for your CUDA version following the
[official instructions](https://pytorch.org/get-started/locally/), then install SIRB:

```bash
git clone https://github.com/SensLiao/Residual_correction_intent.git && cd Residual_correction_intent

pip install -e .
```

A conda environment with every dependency is in [`environment.yml`](environment.yml), and
[`INSTALL.md`](INSTALL.md) lists the environment variables.

## Getting Started

### Data

SIRB uses three things from outside this repository:

| | **What** | **Get it from** | **Put it in** |
| :-: | :-- | :-- | :-- |
| 1 | PSMA-PET-CT-Lesions, NIfTI release v3: 597 PSMA PET/CT examinations of 378 patients with lesion labels | [FDAT](https://doi.org/10.57754/FDAT.g27kx-86t35) | `datasets/PSMA-PET-CT-Lesions/` |
| 2 | Initial segmentations, one mask per examination, from any segmentation model. Ours come from a five-fold [nnU-Net](https://github.com/MIC-DKFZ/nnUNet) `3d_fullres` | made by you, see `datasets/initial_segmentations/README.md` | `datasets/initial_segmentations/` |
| 3 | The official autoPET V simulated user and metrics | [autoPET V](https://github.com/lab-midas/autoPETV), already included | `sirb/third_party/autopet_v/` |

```
datasets/
├── PSMA-PET-CT-Lesions/                    the dataset as downloaded
│   ├── imagesTr/<examination>_0000.nii.gz  CT
│   ├── imagesTr/<examination>_0001.nii.gz  PET
│   └── labelsTr/<examination>.nii.gz       lesion labels
├── initial_segmentations/                  predictions of a segmentation model
│   └── <examination>.nii.gz
└── preprocessed/                           written by tools/prepare_data.py and tools/generate_states.py
```

These are the default folders, and results go to `sirb_results/`. `SIRB_RAW`, `SIRB_INITIAL_SEGMENTATIONS`,
`SIRB_PREPROCESSED` and `SIRB_RESULTS` move them elsewhere. Prepare the examinations of a split once:

```bash
python tools/prepare_data.py --split val
```

### Checkpoint

Put the trained checkpoint at `checkpoints/sirb_psma.pt`. Model weights are not stored in this git repository.

### Correct a segmentation

`SIRBPredictor` keeps the current mask and the interaction history of one examination. Each call to `predict`
corrects the current mask with one scribble and returns the updated mask:

```python
from sirb.utils.config import load_config
from sirb.data.preprocessing import ExaminationCache, cache_directory, pet_reference_from_config
from sirb.interaction.session import Examination
from sirb.sirb_predictor import SIRBPredictor

config = load_config("sirb/configs/eval/five_rounds.json")
predictor = SIRBPredictor.from_config(config, "checkpoints/sirb_psma.pt", mode="standard", device="cuda")

cache = ExaminationCache(cache_directory(config), splits=["val"], pet_reference=pet_reference_from_config(config))
predictor.set_examination(Examination.from_inputs(cache.inputs(<examination id>)))

prediction = predictor.predict(<scribble>, "addition")   # bool ZYX array, drawn on one axial slice
mask = prediction.updated_mask                           # corrected 3D mask, now the current mask
prediction = predictor.predict(<next scribble>, "removal")
```

Use `mode="fast"` for the fast mode (see [Model](#model)).

### Evaluate with the simulated user

In each round the official autoPET V simulated user finds the largest connected component of missed lesion and of
false positive on any axial slice of the current mask, draws a candidate scribble in each in the drawing style of the
examination and keeps the longer one (the addition scribble on a tie). `tools/evaluate.py` corrects every examination of a split and scores the masks of every round by
patient:

```bash
python tools/evaluate.py --config sirb/configs/eval/five_rounds.json --checkpoint checkpoints/sirb_psma.pt \
    --split val --mode standard --device cuda
```

`sirb/configs/eval/ten_rounds.json` runs ten rounds. The test set is read only when `--split test` is given.

### Time per scribble

```bash
python tools/benchmark.py --config sirb/configs/eval/speed.json --checkpoint checkpoints/sirb_psma.pt \
    --mode standard --mode fast --device cuda
```

## Model

![SIRB architecture](assets/architecture.png?raw=true)

A local encoder reads a 336 mm window around the scribble in two streams: the image stream reads PET and CT, and the
state stream reads the current mask, the scribble and the interaction history. A whole-volume encoder adds the
context of the whole examination. The source encoder splits the scribble into at most 16 sources and pools a source
vector for each. The decoder predicts coarse role logits, each source predicts role logits for the target error,
other errors and correct tissue, and native refinement gives every editable voxel on the native PET grid its
role probabilities. The update rule adds or removes, by the correction type, only the editable voxels whose target
probability is at least 0.5.

SIRB has 7.92 M parameters ([config](sirb/configs/model/sirb.json)). The predictor has two modes:

| **Mode** | **Computation** | **Time per scribble (s)** |
| :------- | :-------------- | :-----------------------: |
| `standard` (default) | windows near the scribble, changes connected to the scribble, left-right flip averaging, FP32 | 2.46 |
| `fast` | windows near the scribble, changes connected to the scribble, one pass, bfloat16, `torch.compile` | 0.98 |

Times are medians over 20 scribbles from five validation examinations (the first scribble of each is not counted)
on one NVIDIA RTX 5090 and change with the machine. The scores in Results use every window of the examination
(`predictor.local_update=false` with `--mode standard`). Each option of the two modes can also be set on its own in the `predictor` section of a configuration. The
four ablation variants of the thesis are configurations in `sirb/configs/model/variants/`: No scribble
conditioning, No correct-tissue role, Merged non-target roles and Coarse roles only.

## Dataset and splits

The patient-level split of PSMA-PET-CT-Lesions is in `splits/`:

| **Set** | **Examinations** | **Patients** | **Use** |
| :------ | :--------------: | :----------: | :------ |
| training set | 407 | 264 | training |
| validation set | 99 | 57 | development and timing |
| test set | 91 | 57 | reporting the scores of the frozen model |

`splits/nnunet_folds.json` holds the five patient-level folds of the training and validation examinations on which
the nnU-Net that makes the initial segmentations is trained. `tools/make_splits.py` rebuilds these files from the
dataset and splits another dataset by the same rule.

## Training SIRB

SIRB is trained from scratch for 50,000 training iterations on one GPU, with eight views per iteration drawn from
state pairs and scribble pairs of ground-truth-corrected states and self-generated states. The steps and every
setting are in [`training/README.md`](training/README.md):

```bash
python tools/prepare_data.py --split train
python tools/generate_states.py --split train
python training/train.py --config sirb/configs/train/sirb_psma.json
```

## Repository layout

```
├── sirb/                       library: model and algorithms
│   ├── build_sirb.py           build_sirb(config, checkpoint)
│   ├── sirb_predictor.py       SIRBPredictor: set_examination(), predict(), reset()
│   ├── modeling/               whole-volume encoder, local encoder, source encoder, role decoder, native refinement
│   ├── data/                   PSMA reader, examination cache, states, training pairs, geometry, roles
│   ├── interaction/            simulated user, update rule, correction sessions
│   ├── evaluation/             metrics and patient-level summaries
│   ├── third_party/autopet_v/  official autoPET V scribble simulator and metrics (Apache-2.0)
│   ├── utils/                  configuration, files, checkpoints, logging
│   └── configs/                model, variants, data, training and evaluation configurations
├── training/                   training loop, losses, optimizer, state refresh
├── tools/                      prepare_data, make_splits, generate_states, predict, evaluate, benchmark
├── splits/                     patient-level splits, drawing styles and the nnU-Net folds
├── datasets/                   the dataset and the initial segmentations (not tracked by git)
├── checkpoints/                model weights (not tracked by git)
└── assets/                     figures of this page
```

Every entry point takes `--config`, command-line options and `key=value` overrides of the configuration, and prints
its usage with `--help`.

## License

SIRB is released under the [Apache 2.0 license](LICENSE). The official autoPET V files in
`sirb/third_party/autopet_v/` keep their own Apache 2.0 license.

## Acknowledgements

The simulated user and the DMM metric are the official autoPET V code, included unmodified. We thank the providers
of the PSMA-PET-CT-Lesions dataset (University Hospital, LMU Munich) and the autoPET V organisers.

## Citing SIRB

If you use SIRB in your research, please use the following BibTeX entry.

```bibtex
@mastersthesis{liao2026sirb,
  title  = {Intent-Aware Interactive Correction of Whole-Body {PSMA} {PET/CT} Segmentations},
  author = {Liao, Ruixuan},
  school = {The University of Sydney},
  type   = {Honours thesis},
  year   = {2026}
}
```
