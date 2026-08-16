# Residual Correction Intent

**State-conditioned executable correction programs for interactive whole-body PET/CT lesion segmentation.**

This repository contains the research code for learning, from a sparse signed
scribble and the current segmentation state, a *legal correction program* that
is bound to a specific current-mask component and executed by a constrained
residual editor.

## Research question

A spatial scribble identifies **where** a correction is requested, but not
**which current object** should be edited or **which legal operation** should
be executed. We study whether the system can, without access to ground truth:

1. compile the scribble and current state into an object-referenced legal
   program (`COMPLETE_EXISTING(C2)`, `CREATE_NEW`, `TRIM_LOCAL(C3)`, ...);
2. learn that interpretation **state-relatively** through same-operation
   matched-state supervision (same image, same signed cue, different current
   mask ⇒ different correct program);
3. prove the predicted program actually **governs** downstream correction
   through grammar-legal interventions and a predicted-gold utility gap.

## Method overview

```
PET/CT + current mask M_k + signed scribble S_k
        │
        ▼
Program Compiler (operation authority = cue sign, never predicted)
        │   family scorer (legal families per operation)
        │   + conditional component pointer (ADD existing-object only)
        ▼
Legal program: GROW_LOCAL(Cj) | COMPLETE_EXISTING(Cj) | CREATE_NEW
               TRIM_LOCAL(Cj) | DELETE_COMPONENT(Cj) | REPAIR_OVERSEGMENTED_COMPONENT(Cj)
        │
        ▼
Constrained residual editor (13 channels incl. central selected component)
        │   ADD:  delta ∧ ¬M_k   (monotone union)
        │   REMOVE: delta ∧ C_selected (hard complement protection)
        ▼
M_{k+1}
```

Three data lanes are kept physically separate: an **inference-visible** lane
(PET/CT crops, current mask, cue, component descriptors), a **label-only**
training lane, and an **evaluation/audit-only** lane. Inference first writes an
immutable prediction artifact; an independent evaluator joins predictions to
labels afterwards. The inference loader cannot read either privileged lane.

## Repository layout

```
configs/    frozen v2 experiment config + v3 experiment config (SCEP redesign)
schemas/    frozen v2 intent response schema + v3 program response schema
scripts/
  common/     frozen v2 models/learning (unchanged) + v3 program contract,
              component enumeration, program models, program learning
  data/       materializers (candidates = visible lane, targets = label lane)
  p2t/        v3 program-compiler training entry (J0/J3/J4/J5 arms)
  editor/     v3 program-conditioned editor training entry (J6-J9 ladder)
  evaluation/ v3 evaluator with explicit 2D-plane vs 3D-volume denominator
              domains (never mixed)
tests/      pytest suite for the v3 grammar/components/losses/algebra
protocols/  autoPET V protocol runtime manifest
```

## Status

- v2 (six-class intent ontology + 12-channel editor) is **frozen** and stays
  byte-identical; it is the legacy baseline.
- v3 (SCEP: State-Conditioned Executable Correction Programs) is an audited
  implementation candidate. J0-J9 remain blocked until a newly materialized
  train/validation corpus passes its independent, content-bound receipt; this
  repository asserts no J-series result.
- Data and model weights are **not** included. The dataset is the public
  PSMA-PET/CT whole-body collection (Jeblick et al., Scientific Data 2026);
  the official interactive protocol follows autoPET V.

## Reproducibility and evaluation guardrails

- Splits are patient-disjoint and frozen before training. Locked test images
  are inaccessible without a separate, recorded authorization receipt.
- The primary closed loop advances with the model's actual previous output.
  Teacher-forced correction is reported only as an oracle/conditional ceiling.
- Main-arm comparisons must use the same folds, checkpoint rule, initial mask,
  interaction budget, and patient-clustered uncertainty estimator.
- Undefined patient-by-class metric cells remain undefined and are excluded
  with their support reported; they are never converted into perfect scores.
- Novelty claims are limited to the tested combination of same-operation
  matched-state supervision, object-grounded legal programs, and
  intervention-backed evidence. The code alone does not establish clinical
  safety, superiority, or a field-first claim.

Run the canonical source-only test collection from the repository root:

```bash
python -m pytest -q tests
python -m ruff check scripts tests
```

The schema tests require `jsonschema==4.25.1`. Tests that exercise licensed
checkpoints, vendored upstream repositories, or private runtime assets are
environment-gated and are not evidence of model performance.

## Citation

A manuscript is in preparation; citation will be added on submission.
