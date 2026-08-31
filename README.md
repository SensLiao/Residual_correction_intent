<div align="right"><a href="README.zh-CN.md">简体中文</a></div>

<p align="center"><img src="docs/hero.png" alt="Residual Correction Intent — state-relative correction programs for interactive PET/CT lesion segmentation" width="100%"></p>

<p align="center">
  <img src="https://img.shields.io/badge/status-research%20code-7c3aed?style=flat" alt="Research code">
  <img src="https://img.shields.io/badge/Python-3.10-7c3aed?style=flat" alt="Python 3.10">
  <img src="https://img.shields.io/badge/PyTorch-2.6%20(cu124)-7c3aed?style=flat" alt="PyTorch 2.6, CUDA 12.4">
  <img src="https://img.shields.io/badge/tests-778%20functions-2f9e44?style=flat" alt="778 test functions">
</p>

**Residual Correction Intent** is the research code for the Honours thesis *"From Sparse Corrections to Textual Intent: State-Relative PET/CT Residual Correction"*, by **Ruixuan Liao** (University of Sydney). It studies **state-conditioned executable correction programs (SCEP)** for interactive whole-body PET/CT lesion segmentation, and it is written to be read as **auditable research code**: every contract in the method — grammar legality, data-lane separation, edit guardrails — is enforced by machine-checkable code and tests, not by prose.

<p align="center">
  <a href="#-method">Method</a> ·
  <a href="#-reproducibility--guardrails">Reproducibility</a> ·
  <a href="#-dataset">Dataset</a> ·
  <a href="#-what-you-can-run">Run the tests</a> ·
  <a href="#-citation">Citation</a>
</p>

## 🧭 Overview

**Problem.** In interactive segmentation, a sparse scribble on a predicted mask says **where** a correction is wanted — but not **which object** should be edited, nor **which operation** should run. Existing systems conflate the three: the same scribble might mean "grow this lesion locally", "complete this under-segmented lesion", or "there is a new lesion here", and a free-form network is left to guess while it rewrites the mask.

**Solution.** SCEP separates the three questions and answers each in the right place. The scribble supplies the *location*. A **state-relative program compiler** — reading PET/CT, the current mask, and the scribble, with **no ground-truth access anywhere in this lane** — resolves the *object* and the operand, and deliberately declines to resolve the *operation*: the operation's authority comes from the sign of the cue alone (a positive cue can only `ADD`, a negative cue can only `REMOVE`; the code raises rather than guess if the polarity is ambiguous). What comes out is an object-referenced, grammar-legal program, and only that program is allowed to govern a **constrained residual edit** — so the edit stays bounded instead of letting a network freely repaint the mask.

**Scope.** This repository is the auditable core of in-progress thesis research: contracts, configs, schemas, builders, training/evaluation entry points, and the test suite that enforces them. It **asserts no performance results** — nothing here should be read as evidence the method improves segmentation quality (the confirmatory experiments J0–J9 are registered but not started). The full training environment — CUDA, a vendored nnU-Net tree, the autoPET V package, and the licensed dataset — is not runnable off the original machine; what runs anywhere are the tests and contracts.

## 📌 Contributions (as implemented here)

1. **A state-relative formulation of correction intent** that decomposes a sparse correction into *location* (the scribble), *object* (compiled from the current state), and *operation* (derived from the cue sign, never predicted).
2. **A small, legal correction grammar (v1)** — six program families, schema-enforced, with illegal combinations structurally unrepresentable (e.g. `ADD_NEW_LOCAL` is a forbidden joint goal, and `PREDICT` without an operand cannot validate).
3. **A 13-channel constrained residual editor** whose execution algebra is bounded by construction: `ADD` can only add (`prediction AND NOT current`), `REMOVE` can only remove inside the selected component, and the complement is always protected.
4. **Reproducibility by construction** — three physically separate data lanes, immutable prediction artifacts, exactly-once test-set access ledgers, a 258-entry SHA-256 manifest, and commit-pinned dependencies.
5. **778 test functions across 91 files** that enforce all of the above as executable contracts.

## 🔬 Method

<p align="center"><img src="docs/method.png" alt="Residual Correction Intent method: a scribble compiled into a legal program that governs a constrained residual edit" width="100%"></p>
<p align="center"><sub>The inference-visible lane end to end, with the project's honestly-scoped status and its reproducibility guardrails alongside it.</sub></p>

The inference-visible lane, step by step:

1. **Inputs** — PET and CT context slices, the current mask, and a signed scribble (`FG_POSITIVE` or `BG_NEGATIVE`).
2. **Program compiler** (state-relative, 17-channel input) — enumerates candidate components deterministically from the visible state, scores and **binds** the scribble to an object, and emits a typed program through a fixed trace: `OBSERVE → ENUMERATE → SCORE → BIND → TYPECHECK → PROTECT → EXECUTE`. It may also **abstain** rather than emit an illegal program.
3. **An object-grounded, grammar-legal program** — one of six families (below), carrying its operand and its protected references.
4. **Constrained residual editor** (13-channel) — applies the edit under the execution algebra; the program, not the network, decides what may change.
5. **Output** — the corrected mask.

### The grammar (v1)

| Family | Operation | What it does |
| --- | --- | --- |
| `GROW_LOCAL` | ADD | Grow the bound lesion locally around the cue. |
| `COMPLETE_EXISTING` | ADD | Complete an under-segmented existing lesion. |
| `CREATE_NEW` | ADD | Create a new lesion component at the cue (`NEW_CUE` sentinel — no existing object is bound). |
| `TRIM_LOCAL` | REMOVE | Trim the bound component locally around the cue. |
| `DELETE_COMPONENT` | REMOVE | Delete the bound component entirely. |
| `REPAIR_OVERSEGMENTED_COMPONENT` | REMOVE | Conditional sixth family (config-gated) for over-segmented components. |

The operation column is **never predicted** — `operation_from_cue_sign()` derives it from the scribble's polarity and raises if both or neither polarity is present. Legality is enforced twice: in code ([`scripts/common/petct_program_contract.py`](scripts/common/petct_program_contract.py), the single source of truth) and in schema ([`schemas/petct_program_v1.schema.json`](schemas/petct_program_v1.schema.json), whose conditional clauses tie family→goal, family→operand, and operation→protection).

### The 13 channels

The editor sees exactly: **5 PET slices** (z−2…z+2) + **5 CT slices** + the **current-mask central slice** + the **signed scribble** (one channel, `+1`/`−1` by polarity) + the **selected-component mask** (bitwise zero for `CREATE_NEW` — and contractually never a ground-truth-derived map). The execution algebra then bounds the edit: `ADD = prediction AND NOT current` (new voxels only), `REMOVE = prediction AND selected component` (removal confined to the bound object).

## 🛡 Reproducibility & guardrails

<!-- image-slot: docs/data-lanes.png — the three physically separate data lanes (inference-visible / label-only / audit-only), the allowlist boundary, and the exactly-once test-access ledgers -->

Reproducibility here is not a statement of intent — it is load-bearing structure:

- **Three physically separate data lanes.** Fields are classed *inference-visible*, *label-only*, or *audit-only* in code, and materialisation writes three separate files — `inference.jsonl`, `labels.jsonl`, `audit.jsonl`. The inference lane is guarded by a **14-field allowlist** (not a denylist), and a firewall test asserts that no trajectory id, goal, or case id ever appears in an inference row.
- **Immutable prediction artifacts.** Predictions are written *before* evaluation, so they cannot be tuned to the metric after the fact.
- **Exactly-once test-set access.** Three separate access ledgers (main, M0 baseline, W2.1 official test) are created with `O_EXCL` — opening the test partition a second time fails at the OS level, leaving a permanent record of exactly when test data was touched.
- **Leakage-aware splits.** All splits are at **patient** level (264/57/57 patients, deterministic `stable-patient-hash-v1` with a recorded seed), and evaluation uses patient-clustered bootstrap.
- **An integrity manifest.** [`SHA256SUMS`](SHA256SUMS) pins **258 files** — every script, test, config, schema and protocol (the entire executable surface; only the human-facing docs are outside it).
- **Commit-pinned dependencies.** nnU-Net v2 is pinned to `v2.8.1` at commit `468cf803` *and* to a SHA-256 over its entire runtime tree; the autoPET V scribble simulator is pinned to commit `4a202686` with per-file SHA-256s of the exact three files used; PyTorch is pinned to `2.6.0+cu124`.
- **Frozen prompts and protocols.** Even the VLM probe prompt is a frozen JSON contract with a "retain every attempt; never select only the best response" retry policy.

## 📂 Dataset

| Fact | Value |
| --- | --- |
| Dataset | **PSMA-PET-CT-Lesions v3** (whole-body PSMA PET/CT) |
| Size | **597 cases / 378 patients** (~20.6 GB archive) |
| Licence | **CC BY-NC 4.0** — used under licence, **not redistributed here** |
| Source | [University of Tübingen research data repository](https://fdat.uni-tuebingen.de/) |
| Split | 264 / 57 / 57 patients (train/val/test), patient-level, deterministic seeded hash |
| Scribbles | Simulated via the commit-pinned autoPET V callable (geometry only; polarity assigned by this repo's contract) |

The expected case/patient counts are hard-coded as failing assertions in the dataset builders — a wrong-sized dataset stops the pipeline rather than silently proceeding.

## 🗺 Repository map

| Path | What it holds |
| --- | --- |
| [`configs/`](configs/) | 4 frozen experiment contracts (v2 six-class, v3 SCEP, an operation-control arm, and a 9-method external-comparator contract), each with a `schema_version` and an honest `status` string. |
| [`schemas/`](schemas/) | The two response contracts: intent v2 and program v1 (JSON Schema 2020-12). |
| [`protocols/`](protocols/) | The pinned autoPET V runtime protocol and the frozen VLM probe prompt. |
| [`scripts/common/`](scripts/common/) | The contract layer — grammar, models, learning utilities, freeze and test-access modules. CPU-importable. |
| [`scripts/data/`](scripts/data/) | Manifest builders, lane materialisers, split builders, dataset audits. |
| [`scripts/baseline/`](scripts/baseline/) | The nnU-Net M0 baseline: planning, folds, OOF, validation. |
| [`scripts/p2t/`](scripts/p2t/) · [`scripts/editor/`](scripts/editor/) | Compiler (J0–J5) and editor (J6–J9) training/inference entry points. |
| [`scripts/evaluation/`](scripts/evaluation/) | Metrics (bidirectional, chance-level), aggregation, official-test runners, figure renderers. |
| [`scripts/comparators/`](scripts/comparators/) | Adapters for external interactive-segmentation baselines (nnInteractive, ScribblePrompt, SW-FastEdit, PRISM). |
| [`tests/`](tests/) | 91 test files / 778 test functions — the executable contract suite. |

## 🚀 What you can run

This repository is best read as auditable research code, not a one-command reproduction. What runs on any machine are the **contracts**:

### Requirements

Python 3.10 with `pytest`, `ruff`, `torch`, `numpy`, `scipy`, and `nibabel` installed. No GPU, no dataset, and no nnU-Net checkout are needed for the tests — they run CPU-only against synthetic fixtures.

### Run the contract suite

```bash
python -m pytest -q tests
python -m ruff check scripts tests
```

### What you should see

All tests pass, in the low minutes on CPU. The suite is the specification: among other things it proves the three-lane firewall holds, illegal programs cannot validate, the test-access ledgers are exactly-once, retired entry points stay non-operational, and the frozen v2 baseline remains byte-identical.

The full training/evaluation environment is **not** portable: the setup scripts under [`scripts/setup/`](scripts/setup/) build the original machine's conda + CUDA 12.4 environments and verify the pinned nnU-Net tree, and a number of launch scripts hard-code that machine's paths. This is disclosed rather than hidden — the pins are exact so the environment can be reconstructed deliberately.

## 📊 Project status

In-progress research, honestly scoped. **No performance results are asserted anywhere in this repository.**

| Piece | Where it stands |
| --- | --- |
| v2 six-class baseline | Frozen, byte-identical |
| v3 SCEP program compiler | Audited implementation candidate — code ready, experiments pending |
| Data materialisation | Awaiting an independent receipt |
| Confirmatory experiments J0–J9 | Registered in config, not started |
| Performance results | None asserted |

A manuscript is in preparation; the citation below will be updated on submission.

## 📌 Intended use & limitations

Research code for a thesis in progress — **not a medical device, not for clinical use**, and not a general-purpose segmentation tool. The scope of what has been established is deliberately narrow: implemented and audited contracts, yes; demonstrated clinical or benchmark utility, no.

## 📖 Citation

If you use this research code, please cite it (see [`CITATION.cff`](CITATION.cff) — GitHub's "Cite this repository" button reads it):

```bibtex
@software{liao2026residualcorrectionintent,
  author  = {Liao, Ruixuan},
  title   = {Residual Correction Intent: State-Relative PET/CT Residual Correction},
  year    = {2026},
  url     = {https://github.com/SensLiao/residual-correction-intent},
  note    = {Research code; journal manuscript in preparation}
}
```

## 📄 License

No LICENSE file ships yet — a licence will accompany the manuscript release, and until then the code is **all rights reserved**. The dataset (PSMA-PET-CT-Lesions v3) is licensed **CC BY-NC 4.0** by its authors and is not redistributed here; the pinned upstream components (nnU-Net, autoPET V) remain under their own licences (Apache-2.0).

<p align="center"><sub>Built by <a href="https://github.com/SensLiao">Ruixuan "Sens" Liao</a> · USYD Advanced Computing (Honours)</sub></p>
