# Installation

SIRB needs Python 3.10 and PyTorch 2.4 or later. Training and timed inference need an NVIDIA GPU.

## Environment

```bash
conda env create -f environment.yml
conda activate sirb
```

`environment.yml` installs PyTorch from pip. For a GPU, install the PyTorch build that matches your CUDA version
instead, for example:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

Then, in this folder, install the `sirb` package in editable mode (or add this folder to `PYTHONPATH`):

```bash
pip install -e .
```

## Data and result folders

Three environment variables say where the data and the results are. Only `sirb/configs/paths.json` refers to them;
the data, training and evaluation configurations inherit it.

| Variable | Contents |
|---|---|
| `SIRB_RAW` | `imagesTr/<examination>_0000.nii.gz` (CT), `imagesTr/<examination>_0001.nii.gz` (PET), `labelsTr/` (ground truth), `initial_segmentations/` (the initial segmentation of each examination) |
| `SIRB_PREPROCESSED` | the preprocessed examinations read during training and evaluation, and the stored states used for training |
| `SIRB_RESULTS` | one folder per experiment: `config.json`, `checkpoints/`, `logs/`, `predictions/<split>/`, `scores/<split>/`; default `./sirb_results` |

```bash
export SIRB_RAW=<raw folder>
export SIRB_PREPROCESSED=<preprocessed folder>
export SIRB_RESULTS=<results folder>
```

In PowerShell, write `$env:SIRB_RAW = "<raw folder>"` and so on. A variable that is unset or empty takes the
default given in the configuration; `SIRB_RAW` and `SIRB_PREPROCESSED` have none, so a run that needs them stops
with an error naming the missing variable. Select GPUs with `CUDA_VISIBLE_DEVICES`.

## Check the installation

```bash
python tools/predict.py --help
```

The code was checked with Python 3.10, PyTorch 2.4.1 (CPU), NumPy 1.26.4, SciPy 1.11.4, nibabel 5.4.2,
scikit-image 0.25.2, connected-components-3d 4.1.0, NetworkX 3.4.2 and matplotlib 3.9.2.
