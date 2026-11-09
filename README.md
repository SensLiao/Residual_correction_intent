# SIRB: Intent-Aware Interactive Correction of Whole-Body PSMA PET/CT Segmentations

**[The University of Sydney](https://www.sydney.edu.au/)**

Ruixuan Liao, Jinman Kim

[[`Dataset`](https://doi.org/10.57754/FDAT.g27kx-86t35)] [[`autoPET V`](https://www.autopet.org/autopetv.html)] [[`Simulated user`](https://github.com/lab-midas/autoPETV)] [[`BibTeX`](#citing-sirb)]

![Interactive correction with SIRB](assets/workflow.png?raw=true)

**State-Interventional Referent Binding (SIRB)** corrects an existing 3D lesion segmentation of a whole-body PSMA
PET/CT examination from scribbles. In each round the user reviews the current mask and draws one scribble on an axial
slice: an addition scribble on a missed lesion, or a removal scribble on a false-positive region. SIRB reads the
scribble as a correction intent. It predicts three roles together, the target error that the user selected, other
errors, and correct tissue, and its update rule applies the predicted target error directly as the correction. A
correction therefore follows the error the user chose and keeps what the mask already labels correctly. SIRB has
7.9 M parameters, and in its fast mode it corrects one scribble in 0.98 s (median) on one NVIDIA RTX 5090.

![Correction intent and the three voxel roles](assets/intent_roles.png?raw=true)

*The roles of one addition scribble (a) and one removal scribble (b) on PET. The target error (purple) is the error
the scribble selects. Other errors (yellow hatching) stay for later scribbles, and correct tissue (grey) must not
change.*

## Latest updates

**11/2026 -- Code release**

- The SIRB model, the inference API with an original and a fast mode, five- and ten-round evaluation with the
  autoPET V simulated user, and per-scribble timing.
- Data preparation, training-state generation and the training code.
- The patient-level splits of PSMA-PET-CT-Lesions and the drawing style of every validation and test examination.

## Installation

SIRB needs `python>=3.10` and `torch>=2.4`. Install PyTorch for your CUDA version following the
[official instructions](https://pytorch.org/get-started/locally/), then install SIRB:

```bash
git clone https://github.com/SensLiao/Residual_correction_intent.git && cd Residual_correction_intent

pip install -e .
```

A conda environment with every dependency is in [`environment.yml`](environment.yml). See [`INSTALL.md`](INSTALL.md)
for details.

## Getting Started

### Data folders

SIRB reads the PSMA-PET-CT-Lesions examinations in the layout of the autoPET V release, together with the initial
segmentation that is to be corrected. Three environment variables point to the data and the results:

```
$SIRB_RAW/
├── imagesTr/<examination>_0000.nii.gz        CT
├── imagesTr/<examination>_0001.nii.gz        PET
├── labelsTr/<examination>.nii.gz             ground truth (training and evaluation only)
└── initial_segmentations/<examination>.nii.gz
$SIRB_PREPROCESSED/                           examination cache and training states (written by tools/)
$SIRB_RESULTS/                                one folder per run: configuration, checkpoints, predictions, scores
```

The initial segmentation can come from any automatic model. In our experiments it is the prediction of a five-fold
nnU-Net trained on other patients. Prepare the examinations of a split once:

```bash
export SIRB_RAW=<raw folder> SIRB_PREPROCESSED=<preprocessed folder> SIRB_RESULTS=<results folder>

python tools/prepare_data.py --split val
```

### Checkpoint

Put the trained checkpoint at `checkpoints/sirb_psma.pt` (see [`checkpoints/README.md`](checkpoints/README.md)).
Model weights are not stored in this git repository.

### Correct a segmentation

`SIRBPredictor` keeps the current mask and the interaction history of one examination. Each call to `predict`
corrects the current mask with one scribble and returns the updated mask:

```python
import torch
from sirb.utils.config import load_config
from sirb.data.preprocessing import ExaminationCache, cache_directory, pet_reference_from_config
from sirb.interaction.session import Examination
from sirb.sirb_predictor import SIRBPredictor

config = load_config("sirb/configs/eval/five_rounds.json")
predictor = SIRBPredictor.from_config(config, "checkpoints/sirb_psma.pt", device="cuda")

cache = ExaminationCache(cache_directory(config), splits=["val"], pet_reference=pet_reference_from_config(config))
predictor.set_examination(Examination.from_inputs(cache.inputs(<examination id>)))

prediction = predictor.predict(<scribble>, "addition")   # bool ZYX array, drawn on one axial slice
mask = prediction.updated_mask                           # corrected 3D mask, now the current mask
prediction = predictor.predict(<next scribble>, "removal")
```

The fast mode updates only the windows near the scribble and keeps the changed components connected to it, in one
pass with bfloat16 and `torch.compile`:

```python
predictor = SIRBPredictor.from_config(config, "checkpoints/sirb_psma.pt", device="cuda",
                                      local_update=True, flip_average=False, precision="bf16", compile=True)
```

### Evaluate with the simulated user

In each round the official autoPET V simulated user finds the largest connected component of missed lesion and of
false positive on any axial slice of the current mask, and draws a scribble on the larger one in the drawing style of
the examination: an addition on a missed lesion, a removal on a false positive. `tools/evaluate.py` corrects every
examination of a split for five rounds and scores the masks of every round by patient (final Dice, nAUC-Dice,
nAUC-DMM and false-positive components):

```bash
python tools/evaluate.py --config sirb/configs/eval/five_rounds.json --checkpoint checkpoints/sirb_psma.pt \
    --split val --device cuda
```

`sirb/configs/eval/ten_rounds.json` runs ten rounds. The test set is read only when `--split test` is given.

### Measure the time per scribble

```bash
python tools/benchmark.py --config sirb/configs/eval/speed.json --checkpoint checkpoints/sirb_psma.pt \
    --mode original --mode fast --device cuda
```

## Model Description

![SIRB architecture](assets/architecture.png?raw=true)

A local encoder reads a 336 mm window around the scribble in two streams: the image stream reads PET and CT, and the
state stream reads the current mask, the scribble and the interaction history. A whole-volume encoder adds the
context of the whole examination. The source encoder splits the scribble into at most 16 sources and pools a source
vector for each. The decoder predicts coarse role logits, each source predicts role logits for the target error,
other errors and correct tissue, and native refinement gives every editable voxel at the native PET resolution its
role probabilities. The update rule changes only the voxels whose target probability reaches 0.5, in the direction
of the scribble.

|      **Model**       | **Size (M)** | **Time per scribble, original (s)** | **Time per scribble, fast (s)** | **Final Dice** | **nAUC-Dice** | **nAUC-DMM** |
| :------------------: | :----------: | :---------------------------------: | :-----------------------------: | :------------: | :-----------: | :----------: |
| SIRB <br /> ([config](sirb/configs/model/sirb.json)) | 7.92 | 7.47 | 0.98 | 0.81 | 0.76 | 0.80 |

Time per scribble: median over the 20 warm scribbles on the six timing examinations of the validation set
([`splits/timing.txt`](splits/timing.txt); the first scribble of each examination is left out), measured on an
NVIDIA RTX 5090 with `torch 2.7.1` and CUDA 12.8 and synchronised before and after each call. The original mode computes the whole examination with left-right flip averaging in FP32. Scores are
those of the original mode on the test set, after five rounds from the nnU-Net initial segmentation, averaged by
patient.

The four variants of the ablation study are configuration files in
[`sirb/configs/model/variants/`](sirb/configs/model/variants): No scribble conditioning, No correct-tissue role,
Merged non-target roles and Coarse roles only.

## Results

All methods start from the same nnU-Net initial segmentation and receive the same first scribble. Test set, 82
lesion-positive examinations from 56 patients, five rounds, mean over patients:

| **Method** | **Final Dice** ↑ | **nAUC-Dice** ↑ | **nAUC-DMM** ↑ | **Target correction rate (%)** ↑ | **New error per scribble (mL)** ↓ |
| :--------- | :--------------: | :-------------: | :------------: | :------------------------------: | :-------------------------------: |
| No correction | 0.64 | 0.64 | 0.69 | – | – |
| [2S-ICR](https://doi.org/10.1038/s41598-025-13601-3) | **0.81** | **0.76** | **0.82** | 59.3 | 6.43 |
| [UAM](https://arxiv.org/abs/2608.28461) | 0.79 | 0.73 | 0.81 | 57.4 | 6.05 |
| [nnInteractive](https://arxiv.org/abs/2503.08373) | 0.56 | 0.55 | 0.59 | 52.3 | 49.53 |
| [SegAnyPET-Lesion](https://doi.org/10.1109/ICCV51701.2025.01961) | 0.49 | 0.52 | 0.63 | 43.5 | 23.15 |
| **SIRB (ours)** | **0.81** | **0.76** | 0.80 | **71.3** | **3.27** |

SIRB shares the highest final Dice with 2S-ICR. Each of its scribbles corrects the largest share of the selected
error and introduces the least new error. The target correction rate is the corrected share of the target error
volume, and new error is error that did not exist before the scribble.

![First correction of each method](assets/comparison.png?raw=true)

*The first correction of each method from the same mask and scribble, for one addition (top) and one removal
(bottom). Green and hatched red show the errors left after the correction. On the whole-body projections, purple
marks correct change and orange marks new error.*

![Dice over five correction rounds](assets/dice_over_rounds.png?raw=true)

*Dice averaged by patient over five correction rounds, (a) from the same nnU-Net initial segmentation and (b) from
each method's own initial segmentation. The binary and distance scribble nnU-Nets do not read the current mask and
appear only in (b).*

## Dataset and splits

SIRB is trained and evaluated on
[PSMA-PET-CT-Lesions](https://doi.org/10.57754/FDAT.g27kx-86t35), 597 PSMA PET/CT examinations of 378 patients. The
patient-level split is in [`splits/`](splits/README.md):

| **Set** | **Examinations** | **Patients** | **Use** |
| :------ | :--------------: | :----------: | :------ |
| training set | 407 | 264 | training |
| validation set | 99 | 57 | development, model selection and timing |
| test set | 91 | 57 | reporting the scores of the frozen model |

The drawing style of the simulated user (centerline, random or boundary) is fixed per examination, so every method
sees the same style on the same examination.

## Training SIRB

SIRB is trained from scratch for 50,000 training iterations on one GPU, with eight views per iteration drawn from
state pairs and scribble pairs of three kinds of states: ground-truth-corrected states, states produced by earlier
versions of the model, and self-generated states of the model being trained. The earlier-version states can no
longer be generated, so the released configuration gives their share to the ground-truth-corrected states. See the
[training README](training/README.md) for how to prepare the states and start a run:

```bash
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
├── tools/                      prepare_data, generate_states, predict, evaluate, benchmark
├── splits/                     patient-level splits and drawing styles
├── checkpoints/                model weights (not tracked by git)
└── assets/                     figures of this page
```

Every entry point takes `--config`, command-line options and `key=value` overrides of the configuration, and prints
its usage with `--help`. Configurations are JSON files that inherit with `_base_`.

## License

SIRB is released under the [Apache 2.0 license](LICENSE). The official autoPET V files in
[`sirb/third_party/autopet_v`](sirb/third_party/autopet_v) keep their own Apache 2.0 license.

## Acknowledgements

The simulated user and the DMM metric are the official autoPET V code, included unmodified in
[`sirb/third_party/autopet_v`](sirb/third_party/autopet_v) under its Apache-2.0 license. We thank the providers of
the PSMA-PET-CT-Lesions dataset (University Hospital, LMU Munich) and the autoPET V organisers.

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
