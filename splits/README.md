# Splits

Examination lists of the PSMA PET/CT data (PSMA-PET-CT-Lesions, autoPET V naming
`psma_<patient>_<date>`). An examination is one PET/CT of one patient. Each `.txt` file holds one examination id
per line, sorted. The splits are patient-level: no patient has examinations in two of the training, validation and
test sets.

| File | Examinations | Patients | Use |
|---|---|---|---|
| `train.txt` | 407 | 264 | training set: the only examinations SIRB is trained on |
| `val.txt` | 99 | 57 | validation set: development (model selection, checks, timing) |
| `test.txt` | 91 | 57 | test set: only for reporting the scores of the frozen model |
| `timing.txt` | 6 | 6 | validation examinations on which the time per scribble is measured |
| `val_styles.json` | 99 | | examination → scribble drawing style of the simulated user |
| `test_styles.json` | 91 | | examination → scribble drawing style of the simulated user |

The training and validation sets together hold the 321 patients outside the test set (the training/validation set).
nnU-Net and the task-specific interactive segmentation methods are trained on this pool with patient-level five-fold
cross-validation; SIRB divides it by patient into its training set and validation set.

The evaluation tools read the test set only when `--split test` is given explicitly.

## Drawing styles

The simulated user draws every scribble of an examination in one of the three styles of the official autoPET V
simulator (`centerline`, `random`, `boundary`). The style belongs to the examination, so every method sees the
same style on the same examination. Within each set the assignment follows one deterministic rule: the examinations
are ordered by the SHA-256 digest of `PETCT-OFFICIAL-EQUAL-THIRDS-v1|<examination id>` (ties by the id), and the
styles `centerline`, `random`, `boundary` are dealt in turn along that order. This gives 33 / 33 / 33 validation
examinations and 31 / 30 / 30 test examinations. The result depends on the list the rule is applied to, so the two
files are used as they are and never recomputed on another list.

## Timing

`timing.txt` lists the six validation examinations used to time the correction of one scribble. One further
validation examination, `psma_3959d1c381a5bcd6_2015-11-12`, runs first as a warm-up and is not timed; it is set in
`sirb/configs/eval/speed.json`.

## Provenance

The training and validation lists are the patient-level split that SIRB was trained and developed with. The test
list and its styles are those of the test-set evaluation of SIRB in the thesis, and the validation styles are the
assignment shared by all methods on the validation set. The files hold the same examinations and styles as those
records, sorted by examination id; both style files agree with the rule above for every examination.
