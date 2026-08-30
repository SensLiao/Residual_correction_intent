<div align="right"><a href="README.zh-CN.md">简体中文</a></div>

<p align="center"><img src="docs/hero.png" alt="Residual Correction Intent — state-relative correction programs for interactive PET/CT lesion segmentation" width="100%"></p>

**Residual Correction Intent** is the research code for the Honours thesis *"From Sparse Corrections to Textual Intent: State-Relative PET/CT Residual Correction."* It studies **state-conditioned executable correction programs (SCEP)** for interactive whole-body PET/CT lesion segmentation. The problem it addresses is narrow and concrete: a sparse scribble on a predicted mask says **where** a correction is wanted, but not **which** object should be edited or **which** operation should run. The repository is written by **Ruixuan Liao** (Honours researcher, University of Sydney), also co-author of three 2026 peer-reviewed journal papers in medical imaging and feature selection, and is meant to be read as auditable research code.

## ✨ Highlights

- **Intent, not just location** — a scribble marks where an edit belongs; the compiler resolves *which object* and *which operand* it actually refers to.
- **State-relative program compiler** — it selects the object and the operand but **never predicts the operation**; the operation's authority comes from the sign of the cue.
- **A small, legal grammar (v1)** — every emitted program is drawn from `GROW_LOCAL` / `COMPLETE` / `CREATE_NEW` / `TRIM` / `DELETE` / `REPAIR`.
- **Constrained residual editing** — a 13-channel editor in which `ADD` is a monotone union and `REMOVE` applies hard complement protection.
- **Reproducibility by construction** — three physically separate data lanes, an immutable prediction artifact, leakage-aware splits, a SHA-256 manifest, and commit-pinned dependencies.
- **Extensively tested** — 772 test functions guarding the contracts above.

## 🏗 Method

<p align="center"><img src="docs/method.png" alt="Residual Correction Intent method: scribble compiled into a legal program that governs a constrained residual edit" width="100%"></p>
<p align="center"><sub>The inference-visible lane end to end, with the project's honestly-scoped status and its reproducibility guardrails alongside it.</sub></p>

The method separates three things that interactive correction usually conflates: the *location* a user points at, the *object* that location belongs to, and the *operation* that should be applied to it. A scribble supplies only the first. The compiler resolves the second **state-relatively** — reading the current mask and the state, with **no ground-truth access** at any point in this lane — and deliberately declines to resolve the third, because the operation is already determined by the sign of the cue. What comes out is an object-referenced, grammar-legal program, and only that program is allowed to govern the residual edit, which keeps the edit bounded rather than letting a network freely rewrite the mask.

1. **Inputs** — PET/CT, the current mask, and a signed scribble.
2. **Program Compiler** (state-relative) — selects the object and the operand and emits a program; it never predicts the operation.
3. **Object-grounded legal program** — a grammar-v1 program: `GROW_LOCAL` / `COMPLETE` / `CREATE_NEW` / `TRIM` / `DELETE` / `REPAIR`.
4. **Constrained Residual Editor** (13-channel) — applies the edit under guardrails: `ADD` = monotone union, `REMOVE` = hard complement protection.
5. **Output** — the corrected mask.

## 🔬 Reproducibility & guardrails

- **Three physically separate data lanes** — inference-visible / label-only / evaluation-only.
- **Immutable prediction artifact** — written *before* evaluation, so predictions cannot be tuned to the metric.
- **Leakage-aware evaluation** — patient-clustered bootstrap and leakage-aware splits.
- **Integrity manifest** — a SHA-256 manifest over the tracked artifacts.
- **Pinned dependencies** — nnU-Net v2 and autoPET V pinned by commit.
- **Dataset** — PSMA-PET/CT, 597 cases / 378 patients, CC BY-NC 4.0.

## 🧰 Tech stack

| Area | Technologies |
| --- | --- |
| Language | Python 3.10 |
| Deep learning | PyTorch (torch 2.6), nnU-Net v2 |
| Scientific | NumPy, nibabel, SciPy |
| Data | PSMA-PET/CT (597 cases / 378 patients, CC BY-NC 4.0) |

## 🚀 Getting started

This repository is best read as **auditable research code**, not a one-command reproduction. The full training and evaluation environment — CUDA, a vendored nnU-Net tree, the autoPET V package, and the licensed dataset — is **not runnable off the original machine**. What you *can* run locally are the tests and lint that guard the contracts:

```bash
python -m pytest -q tests
python -m ruff check scripts tests
```

## 🧪 Testing

```bash
python -m pytest -q tests
```

The suite contains **772 test functions** that enforce the data-lane separation, grammar legality, and residual-edit guardrails described above.

## 📌 Limitations

This is in-progress research, and the scope of what has actually been established is deliberately kept narrow. Most importantly, **the repository asserts no performance results** — nothing here should be read as evidence that the method improves segmentation quality. The individual pieces stand as follows:

| Piece | Where it stands |
| --- | --- |
| v2 six-class baseline | Frozen, byte-identical |
| v3 program compiler | Audited implementation candidate |
| Data materialisation | Awaiting an independent receipt |
| J0–J9 experiments | Not started |
| Performance results | None asserted |

A manuscript is in preparation.

## 📄 License

A LICENSE will accompany the manuscript release; until then the code is **all rights reserved** and no `LICENSE` file is present. The dataset (PSMA-PET/CT) is licensed **CC BY-NC 4.0** and is not redistributed here.

<p align="center"><sub>Built by <a href="https://github.com/SensLiao">Ruixuan "Sens" Liao</a> · USYD Advanced Computing (Honours)</sub></p>
