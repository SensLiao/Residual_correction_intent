# Datasets

SIRB needs two things that are not in this repository: the PSMA PET/CT examinations, and an initial segmentation of
every examination for SIRB to correct. Download or make them, and put them in the folders below. These are the
default locations; to keep the data elsewhere, set the environment variables instead (see [INSTALL.md](../INSTALL.md)).

| What | Where to get it | Where to put it | How |
|---|---|---|---|
| **PSMA-PET-CT-Lesions** (NIfTI release, version 3): 597 PSMA PET/CT examinations of 378 patients with expert lesion labels | [FDAT Research Data Repository](https://doi.org/10.57754/FDAT.g27kx-86t35) | `datasets/PSMA-PET-CT-Lesions/` (`$SIRB_RAW`) | [PSMA-PET-CT-Lesions/README.md](PSMA-PET-CT-Lesions/README.md) |
| **Initial segmentations**: one lesion mask per examination, from a five-fold nnU-Net | trained with [nnU-Net v2](https://github.com/MIC-DKFZ/nnUNet) on the folds in [`splits/nnunet_folds.json`](../splits/nnunet_folds.json) | `datasets/PSMA-PET-CT-Lesions/initial_segmentations/` | [initial_segmentations/README.md](PSMA-PET-CT-Lesions/initial_segmentations/README.md) |

Nothing else needs to be downloaded. The evaluation protocol, the simulated user and the metrics of autoPET V, is
included in [`sirb/third_party/autopet_v`](../sirb/third_party/autopet_v/README.md), and the trained SIRB weights go in
[`checkpoints/`](../checkpoints/README.md).

When both are in place, the folder looks like this:

```
datasets/
├── PSMA-PET-CT-Lesions/                         $SIRB_RAW
│   ├── imagesTr/<examination>_0000.nii.gz       CT
│   ├── imagesTr/<examination>_0001.nii.gz       PET
│   ├── labelsTr/<examination>.nii.gz            lesion labels
│   └── initial_segmentations/<examination>.nii.gz
└── preprocessed/                                $SIRB_PREPROCESSED, written by tools/prepare_data.py and tools/generate_states.py
```

An examination is one PET/CT of one patient, named `psma_<patient>_<date>`. The training, validation and test
examinations are listed in [`splits/`](../splits/README.md). Then prepare the examinations of a split:

```bash
python tools/prepare_data.py --split val
```

Only the README files of this folder are tracked by git; the images, labels, masks and the preprocessed data are not.
