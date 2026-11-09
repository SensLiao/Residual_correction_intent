# Initial segmentations

SIRB corrects an existing segmentation. Put one lesion mask per examination in this folder:

```
initial_segmentations/<examination>.nii.gz      binary mask (1 = lesion) on the grid of the PET image
```

Any automatic model can make them. To reproduce the masks of our experiments, train a five-fold
[nnU-Net v2](https://github.com/MIC-DKFZ/nnUNet) as below. Our setting: nnU-Net 2.8.1 with the default `3d_fullres`
configuration and `nnUNetTrainer`, unchanged; the five patient-level folds of
[`splits/nnunet_folds.json`](../../../splits/nnunet_folds.json) over the 506 training and validation examinations
(321 patients); the final checkpoint of each fold.

- **Training and validation examinations**: the prediction of the fold whose validation set holds the examination,
  so no mask comes from a network that was trained on that examination.
- **Test examinations**: the majority vote of the five folds, a voxel is lesion when at least three folds label it.

## 1. Train the five folds

Run from the repository root:

```bash
pip install nnunetv2==2.8.1
export nnUNet_raw=<nnU-Net raw folder> nnUNet_preprocessed=<nnU-Net preprocessed folder> nnUNet_results=<nnU-Net results folder>

mkdir -p $nnUNet_raw/Dataset901_PSMA
ln -s "$(pwd)/datasets/PSMA-PET-CT-Lesions/imagesTr" $nnUNet_raw/Dataset901_PSMA/imagesTr    # or copy the folder
ln -s "$(pwd)/datasets/PSMA-PET-CT-Lesions/labelsTr" $nnUNet_raw/Dataset901_PSMA/labelsTr
cat > $nnUNet_raw/Dataset901_PSMA/dataset.json << 'EOF'
{"channel_names": {"0": "CT", "1": "PET"}, "labels": {"background": 0, "tumor": 1},
 "numTraining": 597, "file_ending": ".nii.gz"}
EOF

nnUNetv2_plan_and_preprocess -d 901 -c 3d_fullres --verify_dataset_integrity
cp splits/nnunet_folds.json $nnUNet_preprocessed/Dataset901_PSMA/splits_final.json
for fold in 0 1 2 3 4; do nnUNetv2_train 901 3d_fullres $fold; done
```

The channel names make nnU-Net normalise channel 0 with its CT scheme and channel 1 with z-scores. The test
examinations are in `imagesTr/` as well, but no fold trains or validates on them.

## 2. Training and validation examinations

At the end of its training each fold predicts its own validation examinations into `fold_<k>/validation/`:

```bash
cp $nnUNet_results/Dataset901_PSMA/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_*/validation/*.nii.gz \
   datasets/PSMA-PET-CT-Lesions/initial_segmentations/
```

## 3. Test examinations

Predict the test examinations with each fold, then take the majority vote:

```bash
mkdir -p imagesTs
while read id; do cp datasets/PSMA-PET-CT-Lesions/imagesTr/${id}_000[01].nii.gz imagesTs/; done < splits/test.txt
for fold in 0 1 2 3 4; do
  nnUNetv2_predict -i imagesTs -o test_fold$fold -d 901 -c 3d_fullres -f $fold -chk checkpoint_final.pth
done

python - << 'EOF'
from pathlib import Path
import nibabel as nib
import numpy as np

out = Path("datasets/PSMA-PET-CT-Lesions/initial_segmentations")
for case in sorted(Path("test_fold0").glob("psma_*.nii.gz")):
    folds = [nib.load(f"test_fold{k}/{case.name}") for k in range(5)]
    votes = sum((np.asanyarray(f.dataobj) > 0).astype(np.uint8) for f in folds)
    nib.save(nib.Nifti1Image((votes >= 3).astype(np.uint8), folds[0].affine, folds[0].header), out / case.name)
EOF
```

## Optional probability

`tools/prepare_data.py` checks that every mask has the grid of its PET image. A file `<examination>.npz` next to the
mask, holding the lesion probability under the key `foreground_probability` in the voxel order of the NIfTI file, is
optional: it is checked against the mask (probability above 0.5) and stored with the examination, and SIRB itself
reads only the mask.
