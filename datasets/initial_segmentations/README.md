# Initial segmentations

SIRB corrects an existing segmentation. Put one lesion mask per examination here. This folder is the default
initial segmentation folder (`SIRB_INITIAL_SEGMENTATIONS`).

```
datasets/initial_segmentations/
├── <examination>.nii.gz      binary mask (1 = lesion) on the voxel grid of the examination's PET image
└── <examination>.npz         optional: the lesion probability of the model that made the mask
```

The masks can come from any segmentation model, and SIRB corrects them without retraining. `tools/prepare_data.py`
checks that every mask has the voxel grid of its PET image. The optional `<examination>.npz` holds the lesion
probability under the key `foreground_probability`, in the voxel order of the NIfTI file. When it is present, it is
checked against the mask (probability above 0.5) and stored with the examination. SIRB itself reads only the mask.

## Our initial segmentations: five-fold nnU-Net

In our experiments the masks come from [nnU-Net](https://github.com/MIC-DKFZ/nnUNet) 2.8.1 with the default
`3d_fullres` configuration and `nnUNetTrainer`, unchanged, trained with five patient-level folds over the 506
training and validation examinations (321 patients) of `splits/nnunet_folds.json`. Each fold uses its final
checkpoint.

- **Training and validation examinations**: the prediction of the fold that holds the examination in its
  validation set (an out-of-fold prediction), so no mask comes from a network that was trained on that examination.
- **Test examinations**: the majority vote of the five folds. A voxel is lesion when at least three of the five
  folds label it.

The test examinations are in no fold. The commands below run from the repository root.

### 1. Train the five folds

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

### 2. Training and validation examinations

At the end of its training each fold predicts its own validation examinations into `fold_<k>/validation/`:

```bash
mkdir -p datasets/initial_segmentations
cp $nnUNet_results/Dataset901_PSMA/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_*/validation/*.nii.gz \
   datasets/initial_segmentations/
```

### 3. Test examinations

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

out = Path("datasets/initial_segmentations")
for case in sorted(Path("test_fold0").glob("psma_*.nii.gz")):
    folds = [nib.load(f"test_fold{k}/{case.name}") for k in range(5)]
    votes = sum((np.asanyarray(f.dataobj) > 0).astype(np.uint8) for f in folds)
    nib.save(nib.Nifti1Image((votes >= 3).astype(np.uint8), folds[0].affine, folds[0].header), out / case.name)
EOF
```
