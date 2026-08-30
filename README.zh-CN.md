<div align="right"><a href="README.md">English</a></div>

<p align="center"><img src="docs/hero.png" alt="Residual Correction Intent banner" width="100%"></p>

<p align="center"><strong>把一笔稀疏涂鸦，编译为面向对象、且符合文法的修正程序，用于交互式全身 PET/CT 病灶分割。</strong></p>

<p align="center">
  <img src="https://img.shields.io/badge/status-research%20code-a78bfa?style=flat-square" alt="Research code">
  <img src="https://img.shields.io/badge/PyTorch-2.6-a78bfa?style=flat-square" alt="PyTorch 2.6">
  <img src="https://img.shields.io/badge/nnU--Net-v2-a78bfa?style=flat-square" alt="nnU-Net v2">
  <img src="https://img.shields.io/badge/Python-3.10-a78bfa?style=flat-square" alt="Python 3.10">
  <img src="https://img.shields.io/badge/tests-772-a78bfa?style=flat-square" alt="772 tests">
  <img src="https://img.shields.io/badge/license-all%20rights%20reserved-a78bfa?style=flat-square" alt="All rights reserved">
</p>

## 概览

**Residual Correction Intent** 是 Honours 毕业论文 *"From Sparse Corrections to Textual Intent: State-Relative PET/CT Residual Correction"* 的研究代码。它研究用于交互式全身 PET/CT 病灶分割的 **state-conditioned executable correction programs（SCEP）**。作者：**Ruixuan Liao**（悉尼大学 Honours 研究者），同时也是三篇 2026 年医学影像与特征选择方向同行评审期刊论文的合著者。

核心思想：一笔稀疏涂鸦只说明**在哪里**（where）需要修正，却没有说明要编辑**哪个对象**（which object）、要执行**哪种操作**（which operation）。系统将该涂鸦、当前掩码（mask）与状态一起，编译成一个面向对象、且符合文法的修正程序 —— 该程序以**状态相对（state-relative）、且不访问 ground-truth** 的方式学习 —— 随后由这个程序来支配一次受约束的残差编辑（residual edit）。

## ✨ 亮点

- **意图，而不仅是位置** —— 涂鸦标出「在哪里」；编译器负责解析它所指的*哪个对象*与*哪个操作数*。
- **状态相对的程序编译器** —— 选择对象与操作数，但**从不预测操作（operation）**；操作的依据来自提示（cue）的符号。
- **一套小而合法的文法（grammar v1）** —— 程序取自 `GROW_LOCAL` / `COMPLETE` / `CREATE_NEW` / `TRIM` / `DELETE` / `REPAIR`。
- **受约束的残差编辑** —— 一个 13 通道（13-channel）编辑器，其中 `ADD` 为单调并集（monotone union），`REMOVE` 施加硬补集保护（hard complement protection）。
- **从构造上保证可复现** —— 三条物理隔离的数据通道、一个不可变的预测工件（immutable prediction artifact）、防泄漏的数据划分、一份 SHA-256 清单，以及按 commit 固定的依赖（详见下文）。
- **充分测试** —— 772 个测试函数守护上述各项契约。

## 🏗 工作原理

<p align="center"><img src="docs/method.png" alt="Residual Correction Intent method diagram" width="100%"></p>

该流程把一个提示（cue）转化为一次有界的编辑：

1. **输入** —— PET/CT、当前掩码，以及一笔带符号的涂鸦（signed scribble）。
2. **Program Compiler**（状态相对） —— 选择对象与操作数并生成程序；它**从不预测操作**（操作的依据来自提示符号）。
3. **面向对象的合法程序** —— 一个 grammar-v1 程序：`GROW_LOCAL` / `COMPLETE` / `CREATE_NEW` / `TRIM` / `DELETE` / `REPAIR`。
4. **Constrained Residual Editor**（13 通道） —— 在护栏下施加编辑：`ADD` = 单调并集，`REMOVE` = 硬补集保护。
5. **输出** —— 修正后的掩码。

## 🔬 可复现性与护栏

- **三条物理隔离的数据通道** —— inference-visible / label-only / evaluation-only。
- **不可变的预测工件** —— 在评估**之前**写入，因此预测无法被调向指标（metric）。
- **防泄漏的评估** —— 按病人聚类的 bootstrap，以及防泄漏的数据划分。
- **完整性清单** —— 对受追踪工件的一份 SHA-256 清单。
- **固定的依赖** —— nnU-Net v2 与 autoPET V 均按 commit 固定。
- **数据集** —— PSMA-PET/CT，597 例 / 378 位病人，CC BY-NC 4.0。
- **测试套件** —— 772 个测试函数。

## 🧰 技术栈

| 领域 | 技术 |
| --- | --- |
| 语言 | Python 3.10 |
| 深度学习 | PyTorch（torch 2.6）、nnU-Net v2 |
| 科学计算 | NumPy、nibabel、SciPy |
| 数据 | PSMA-PET/CT（597 例 / 378 位病人，CC BY-NC 4.0） |

## 🚀 快速开始

本仓库更适合作为**可审计的研究代码**（auditable research code）来阅读，而不是一条命令即可复现的项目。完整的训练与评估环境 —— CUDA、内置（vendored）的 nnU-Net 目录树、autoPET V 包，以及受许可的数据集 —— **无法在原始机器之外运行**。你在本地*可以*运行的，是守护这些契约的测试与静态检查（lint）：

```bash
python -m pytest -q tests
python -m ruff check scripts tests
```

## 🧪 测试

```bash
python -m pytest -q tests
```

该套件包含 **772 个测试函数**，用于强制执行上文所述的数据通道隔离、文法合法性以及残差编辑护栏。

## 📌 项目状态

这是一份诚实、进行中的研究 —— 每个部分的状态都被刻意追踪：

- **v2 六分类基线 —— 已冻结（FROZEN）**（逐字节一致，byte-identical）。
- **v3 程序编译器 —— 经过审计的实现候选（audited implementation candidate）。**
- **数据物化（materialisation） —— 等待一份独立回执（independent receipt）。**
- **J0–J9 实验 —— 尚未开始。**
- **性能结果 —— 无。** 本仓库**不主张**任何性能结果。

一篇手稿正在准备中。

## 📄 许可证

LICENSE 将随手稿发布一同提供；在此之前代码为 **all rights reserved（保留所有权利）**（目前尚无 `LICENSE` 文件）。后续将添加 `CITATION.cff`。数据集（PSMA-PET/CT）以 **CC BY-NC 4.0** 许可，本仓库不对其进行再分发。

<p align="center"><sub>Built by <a href="https://github.com/SensLiao">Ruixuan "Sens" Liao</a> · USYD Advanced Computing (Honours)</sub></p>
