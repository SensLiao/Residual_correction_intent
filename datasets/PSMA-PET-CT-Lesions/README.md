# PSMA-PET-CT-Lesions

Put the PSMA PET/CT examinations here. This folder is the default `$SIRB_RAW`.

**Download.** Get the NIfTI release, version 3, of PSMA-PET-CT-Lesions from the
[FDAT Research Data Repository](https://doi.org/10.57754/FDAT.g27kx-86t35) (University Hospital, LMU Munich;
distributed by the University of Tübingen). It holds 597 PSMA PET/CT examinations of 378 patients with expert lesion
labels. The release unpacks into a folder `PSMA-PET-CT-Lesions_v3/`: move its contents here, so that `imagesTr/` and
`labelsTr/` sit directly in this folder (or set `SIRB_RAW` to the unpacked folder instead).

**Expected layout.**

```
datasets/PSMA-PET-CT-Lesions/
├── imagesTr/                 1,194 files, two per examination
│   ├── <examination>_0000.nii.gz        CT
│   └── <examination>_0001.nii.gz        PET
├── labelsTr/                 597 files: <examination>.nii.gz, the lesion labels
├── initial_segmentations/    the masks SIRB corrects, see initial_segmentations/README.md
└── dataset.json, splits_final.json, psma_metadata.csv    part of the release; SIRB does not read them
```

An examination is named `psma_<patient>_<date>`. The release is already in the nnU-Net raw layout, so the same files
also train the nnU-Net that makes the initial segmentations. SIRB uses its own patient-level split in
[`splits/`](../../splits/README.md), with 91 test examinations held out from every training run, not the
`splits_final.json` of the release.

The dataset comes with its own licence and citation; see its record. The original DICOM collection is on
[TCIA](https://www.cancerimagingarchive.net/collection/psma-pet-ct-lesions/).
