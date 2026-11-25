# PSMA-PET-CT-Lesions

Put the PSMA PET/CT examinations here. This folder is the default dataset folder (`SIRB_RAW`).

**Download.** Get the NIfTI release, version 3, of PSMA-PET-CT-Lesions from the FDAT Research Data Repository:
<https://doi.org/10.57754/FDAT.g27kx-86t35>. It holds 597 PSMA PET/CT examinations of 378 patients with expert
lesion labels. The release unpacks into a folder `PSMA-PET-CT-Lesions_v3/`. Move its `imagesTr/` and `labelsTr/`
here, or set `SIRB_RAW` to the unpacked folder instead.

**Layout.**

```
datasets/PSMA-PET-CT-Lesions/
├── imagesTr/                            1,194 files, two per examination
│   ├── <examination>_0000.nii.gz        CT (HU)
│   └── <examination>_0001.nii.gz        PET (SUV)
└── labelsTr/                            597 files
    └── <examination>.nii.gz             lesion labels (1 = lesion)
```

An examination is one PET/CT of one patient, named `psma_<patient>_<date>`. The CT, the PET and the labels of an
examination share one voxel grid. The other files of the release (`dataset.json`, `splits_final.json`,
`psma_metadata.csv`) are not read.

The dataset comes with its own licence and citation, given on its FDAT record.
