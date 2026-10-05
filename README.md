# When Pruning Meets Interpretability: Preserving Sparse Autoencoder Robustness in LLMs

[![COLM 2026](https://img.shields.io/badge/COLM-2026-blue)](https://colmweb.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)

Official implementation of **"When Pruning Meets Interpretability: Preserving Sparse Autoencoder Robustness in LLMs"** (COLM 2026).

**Suchit Gupte, Xueru Zhang, Mohammad Mahdi Khalili**
Department of Computer Science, The Ohio State University

---

The original repository is [osu-srml/sae-robustness-under-pruning](https://github.com/osu-srml/sae-robustness-under-pruning).

## 📖 Overview

Sparse autoencoders (SAEs) are widely used to interpret the internal representations of large language models, yet their reliability under post-hoc model compression remains poorly understood. This repository provides a systematic study of how weight pruning affects SAE behavior.

**Key contributions:**

- 🧮 **Perturbation-theoretic framework**: We show that, for a fixed SAE, pruning-induced degradation is governed by **perturbation energy** — a covariance-weighted norm ε² = tr(ΔW Σ ΔWᵀ) — modulated by the SAE's Lipschitz constant.
- 📉 **Formal separation of pruning methods**: Magnitude, Wanda, and SparseGPT each minimize a mathematically distinct approximation of ε², forming a natural hierarchy:

  ```
  tr(ΔW I ΔWᵀ)  ≺  tr(ΔW diag(Σ) ΔWᵀ)  ≺  tr(ΔW Σ ΔWᵀ)
    Magnitude          Wanda                SparseGPT
  ```

  Magnitude pruning ignores activation geometry and degrades SAEs most severely; activation-aware methods (Wanda, SparseGPT) implicitly control ε² and are substantially more robust.
- 🏗️ **Structural vulnerability**: Middle layers are consistently more sensitive to pruning than early or late layers — across all methods, metrics, and sparsity levels — driven by accumulated upstream perturbation rather than local ε².
- ⚖️ **Layer-wise sparsity allocation**: A ramp schedule that protects early layers achieves lower perplexity than uniform sparsity at the same average compression budget.

## 🔬 Main Results

### Method ordering (gemma-2-2b, 50% sparsity)

The ordering **SparseGPT ≻ Wanda ≻ Magnitude** holds across all eight SAEBench metrics and all 26 layers:

| Metric | Magnitude | Wanda | SparseGPT |
|---|---|---|---|
| Worst-case SCR degradation | up to −60% | within 15–25% of baseline | within 15–25% of baseline |
| Worst-case TPP degradation | up to −80% | within 15–25% of baseline | within 15–25% of baseline |

⚠️ **Aggregate metrics hide damage**: Magnitude-pruned middle layers keep KL divergence scores ≥ 0.96 while SCR and TPP collapse — aggregate reconstruction fidelity alone cannot certify SAE reliability.

### Layer-wise sparsity allocation (perplexity, gemma-2-2b @ 50% avg. sparsity)

| Method | Uniform | Layer-wise |
|---|---|---|
| Wanda | 212 | **148** |
| SparseGPT | 141 | **98** |

The ramp schedule assigns 20% sparsity to early layers, increasing linearly to 60% at the middle layers and holding through the last layer.

### Cross-model generalization

The theoretical ordering is validated on four model–SAE pairs spanning ~2 orders of magnitude in parameter count:

| Model | SAE Source |
|---|---|
| `pythia-70m` | SAELens |
| `gemma-2-2b` | Gemma Scope (all residual layers 0–25) |
| `gemma-2-9b` | Gemma Scope |
| `mistral-7b` | Mistral-7B Residual Stream SAEs |

## 🚀 Getting Started

### Installation

```bash
git clone https://github.com/KhaliliMahdi/sae-robustness-under-pruning.git
cd sae-robustness-under-pruning
pip install -r requirements.txt
```

Key dependencies: `torch`, `transformers`, `sae-lens`, `sae-bench`.

### Pruning a model

Apply one-shot pruning (no retraining) to a dense checkpoint:

```bash
python prune.py \
    --model google/gemma-2-2b \
    --method wanda \            # magnitude | wanda | sparsegpt
    --sparsity 0.5 \
    --calib-data openwebtext \
    --calib-samples 128 \
    --output-dir checkpoints/gemma-2-2b-wanda-50
```

### Layer-wise sparsity schedule

```bash
python prune.py \
    --model google/gemma-2-2b \
    --method sparsegpt \
    --schedule ramp \
    --sparsity-min 0.2 \
    --sparsity-mid 0.6 \
    --output-dir checkpoints/gemma-2-2b-sparsegpt-ramp
```

### Evaluating SAE robustness

Run the four SAEBench metric categories (Core, Feature Absorption, SCR, TPP) on every residual-stream layer of a pruned model:

```bash
python evaluate.py \
    --model-checkpoint checkpoints/gemma-2-2b-wanda-50 \
    --sae-release gemma-scope-2b-pt-res \
    --metrics core absorption scr tpp \
    --scr-tpp-threshold 10 \
    --output results/gemma-2-2b-wanda-50.json
```

### Measuring perturbation energy

Compute the empirical perturbation energy ε² = E[‖ΔW x‖²] per layer:

```bash
python perturbation_energy.py \
    --dense-model google/gemma-2-2b \
    --pruned-model checkpoints/gemma-2-2b-wanda-50 \
    --calib-data openwebtext
```

> **Note:** Script names and flags above reflect the intended workflow; see each script's `--help` for the full option list.

## 📊 Evaluation Suite

We validate against four SAEBench metric categories, which jointly cover the full stack of SAE failure modes:

| Failure Mode | Metric | What it detects |
|---|---|---|
| Reconstruction degrades | **Core** (KL score, CE score, explained variance, cosine sim, MSE) | Aggregate reconstruction no longer preserves model behavior |
| Features break | **Feature Absorption** | Individual latents go silent or fire spuriously despite good aggregate scores |
| Interventions stop working | **SCR** | Latent ablations lose their intended causal effect |
| Interventions lose selectivity | **TPP** | Ablations damage concepts they were never meant to affect |

Notes on coverage: Feature Absorption is unreliable for models under ~2B parameters, so it is excluded for `pythia-70m`; `mistral-7b` has pretrained SAEs for only one layer per depth group.

## 📁 Repository Structure

```
.
├── prune.py                  # One-shot pruning (magnitude / wanda / sparsegpt)
├── evaluate.py               # SAEBench evaluation on pruned checkpoints
├── perturbation_energy.py    # Empirical ε² measurement
├── schedules/                # Uniform and layer-wise sparsity schedules
├── analysis/                 # Plotting: per-layer profiles, heatmaps, threshold sweeps
├── results/                  # Raw metric outputs
└── requirements.txt
```

## 📚 Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{gupte2026pruning,
  title     = {When Pruning Meets Interpretability: Preserving Sparse Autoencoder Robustness in {LLM}s},
  author    = {Gupte, Suchit and Zhang, Xueru and Khalili, Mohammad Mahdi},
  booktitle = {Conference on Language Modeling (COLM)},
  year      = {2026}
}
```

## 🙏 Acknowledgments

This work was funded in part by the National Science Foundation under award numbers IIS-2202699, IIS-2416895, IIS-2301599, CMMI-2301601, and DMS-2529302.

We build on the following open resources:
- [SAELens](https://github.com/decoderesearch/SAELens) — pretrained SAE checkpoints
- [SAEBench](https://arxiv.org/abs/2503.09532) — evaluation suite
- [Gemma Scope](https://arxiv.org/abs/2408.05147) — SAEs for Gemma 2
- [Wanda](https://arxiv.org/abs/2306.11695) and [SparseGPT](https://arxiv.org/abs/2301.00774) — pruning methods

## 📬 Contact

For questions, please open an issue or contact:
- Suchit Gupte — `gupte.31@osu.edu`
- Xueru Zhang — `zhang.12807@osu.edu`
- Mohammad Mahdi Khalili — `khalili.17@osu.edu`
