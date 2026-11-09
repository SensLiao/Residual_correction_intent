# Splits

Examination lists of the PSMA PET/CT data (PSMA-PET-CT-Lesions, autoPET V naming `psma_<patient>_<date>`). An
examination is one PET/CT of one patient. Each `.txt` file holds one examination id per line, sorted. The splits are
patient-level: no patient has examinations in two of the training, validation and test sets.

| File | Examinations | Patients | Use |
|---|---|---|---|
| `train.txt` | 407 | 264 | training set: the only examinations SIRB is trained on |
| `val.txt` | 99 | 57 | validation set: development (model selection, checks, timing) |
| `test.txt` | 91 | 57 | test set: only for reporting the scores of the frozen model |
| `timing.txt` | 6 | 6 | validation examinations on which the time per scribble is measured |
| `val_styles.json` | 99 | | examination → scribble drawing style of the simulated user |
| `test_styles.json` | 91 | | examination → scribble drawing style of the simulated user |
| `nnunet_folds.json` | 506 | 321 | five folds over the training and validation examinations, for the nnU-Net that makes the initial segmentations (nnU-Net `splits_final.json` format) |

These are the files of our experiments. The evaluation tools read the test set only when `--split test` is given.

## How the splits are made

`tools/make_splits.py` writes every file except `timing.txt` from the examination ids alone, with three fixed rules
(`sirb.data.psma`):

1. **Training, validation and test sets** (`split_patients`). The patient id of an examination is its id without the
   date. The patients are ordered by the SHA-256 digest of `PETCT-STABLE-PATIENT-HASH-v1|<seed>|<patient id>`
   (patient id in lower case, ties by the id), and the first 264, the next 57 and the last 57 patients form the
   training, validation and test sets, with all their examinations. Our seed is 20260717.
2. **Drawing styles** (`assign_drawing_styles`). The simulated user draws every scribble of an examination in one of
   the three styles of the official autoPET V simulator (`centerline`, `random`, `boundary`). The style belongs to
   the examination, so every method sees the same style on the same examination. Within the validation set, and
   within the test set, the examinations are ordered by the SHA-256 digest of
   `PETCT-OFFICIAL-EQUAL-THIRDS-v1|<examination id>` (ties by the id) and the three styles are dealt in turn along
   that order. This gives 33 / 33 / 33 validation and 31 / 30 / 30 test examinations.
3. **nnU-Net folds** (`nnunet_folds`). The 321 training and validation patients are ordered by the SHA-256 digest of
   `PETCT-M0-EXCLTEST-FOLD-v1|<patient id>`, and each in turn joins the fold that holds the fewest examinations so
   far (the lowest fold on ties). Fold *k* validates on the examinations of its patients and trains on the others,
   so every training and validation examination is validated in exactly one fold, and no test examination is in any
   fold. The folds hold 104, 102, 100, 100 and 100 validation examinations.

On the 597 examinations of PSMA-PET-CT-Lesions the script reproduces the six files here byte for byte:

```bash
python tools/make_splits.py --output my_splits
```

It reads the examination ids from the dataset's `labelsTr/` (`SIRB_RAW`), or from a list with
`--examinations <file>` (one id per line). For another dataset, set the number of patients in each set with
`--patients TRAIN VAL TEST` (they must add up to all patients), the seed with `--seed` and the number of folds with
`--folds`. A file that already exists in `--output` (default `splits/`) is replaced only with `--overwrite`.

nnU-Net and the task-specific interactive segmentation methods we compare with are trained with the five folds of
`nnunet_folds.json` on the 506 training and validation examinations. SIRB is trained on the training set and
developed on the validation set.

## Timing

`timing.txt` lists the six validation examinations used to time the correction of one scribble. One further
validation examination, `psma_3959d1c381a5bcd6_2015-11-12`, runs first as a warm-up and is not timed. Both are set
in `sirb/configs/eval/speed.json`.
