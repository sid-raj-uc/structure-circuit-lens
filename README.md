# Graphical Model Structure Learning for Sparse Feature Circuits

**TTIC 31180 — Probabilistic Graphical Models · Final Project**  
Siddharth Raj · sidraj@uchicago.edu

---

## Overview

Can classical PGM structure learning recover the circuits inside a language model?

This project applies three structure-learning methods — PC (Fisher-Z and G-test variants), GES-BIC, and Mean Ablation Patching — to the sparse feature activations of GPT-2 Small, extracted via Sparse Autoencoders (SAEs). We learn a directed graph over 76 SAE features (30 from layer 1, 46 from layer 2) and validate recovered edges using two complementary ground truths: direct activation patching (interventional) and Spearman rank correlation (observational).

---

## Background

**Sparse Autoencoders (SAEs)** decompose GPT-2's dense residual stream into an overcomplete dictionary of interpretable features. Each feature is approximately monosemantic — it activates for one concept rather than many (solving polysemanticity).

We tap the residual stream at:
- `blocks.1.hook_resid_pre` → **30 L1 features** (input to layer 1)
- `blocks.2.hook_resid_pre` → **46 L2 features** (input to layer 2)

These layers sit directly on the **induction circuit** (Olsson et al. 2022): prev-token heads in L0 → L1 residual → induction heads in L1 → L2 residual. This gives us a known ground truth to validate against.

---

## Methods

### 1. PC Algorithm (Constraint-Based)

Runs the skeleton phase of PC with two conditional independence tests:

| Variant | Data | Test |
|---|---|---|
| PC-FisherZ | Continuous activations | Partial correlation (Fisher-Z transform) |
| PC-GSq | Binary activations (fired / not fired) | G-test (chi-squared) |

- Depth limited to `max_depth=3` for tractability
- **Orientation**: uses architectural causal ordering (L1 precedes L2 in the forward pass) instead of Meek's rules — valid and stronger given known layer ordering
- Tested at α ∈ {0.01, 0.05, 0.10}

### 2. GES-BIC (Score-Based)

Custom fast implementation of Greedy Equivalence Search:

- **Score**: BIC with penalty `log(n)` per edge
- **Forward phase**: greedily add the single edge with highest ΔBIC
- **Backward phase**: greedily remove edges that improve BIC
- **Key optimisation**: precomputes `C = Xᵀ X` once; all score evaluations use Schur complements — no re-inversion during search
- Runtime: ~36s for p=76, N=5000 (vs hours for causal-learn's GES)

### 3. Mean Ablation Patching (Causal Proxy)

For each L1 feature i: replace its activation with the dataset mean, measure mean |Δ| in each L2 feature j over all 5000 prompts. Top 5% of scores (95th percentile threshold) form the edge set.

> **Note**: this is mean-ablation sensitivity, not Marks et al.'s integrated-gradient circuit-finding method.

---

## Validation

### Interventional Ground Truth (Zero-Ablation)
Zero-ablates each L1 feature, checks if the target L2 feature changes by >10% of its mean clean activation over 500 held-out prompts. Confirms direct causal influence.

### Observational Ground Truth (Spearman Correlation)
Spearman rank correlation between L1 and L2 feature activations across all 5000 prompts. Confirmed if |ρ| > 0.20 and p < 0.01. Bias-free — shares no inductive bias with any method.

> **Bias note**: ablation-based validation structurally favours Mean Ablation Patching (both use disruption). Spearman IA gives a fair cross-method comparison.

40 edges were stratified-sampled (20 PGM-only, 10 ablation-only, 10 agreement) and validated manually.

---

## Results

| Method | Edges | Ablation IA | Runtime |
|---|---|---|---|
| PC-FisherZ α=0.01 | 184 | 0.667 | 33s |
| PC-FisherZ α=0.05 | 223 | 0.619 | 64s |
| PC-FisherZ α=0.10 | 255 | 0.619 | 98s |
| PC-GSq α=0.01 | 160 | 0.643 | 247s |
| PC-GSq α=0.05 | 200 | 0.611 | 499s |
| PC-GSq α=0.10 | 226 | 0.600 | 682s |
| **GES-BIC** | **157** | **0.667** | **36s** |
| Mean Ablation 95th | 69 | 0.850* | — |

*Ablation IA for Mean Ablation is biased upward — see bias note above.

**GES-BIC** achieves the best precision-speed trade-off: fewest edges among PGM methods, same Ablation IA as PC's best, in 36 seconds.

**Pairwise Jaccard** between PGM methods and Mean Ablation ≈ 0.10 — they find largely non-overlapping edge sets, confirming they measure fundamentally different things (statistical dependence vs. interventional sensitivity).

---

## Head-Level Analysis

SAE features are projected back to attention heads via `W_O`:

```
attr[h, f] = mean( head_out[:, h, :] @ W_enc[:, feature_f] )
```

Each feature is assigned to its dominant head (argmax of |attr| over 12 heads). Feature-level edges are then aggregated to a 12×12 head-level matrix.

**Result**: The known induction circuit (L0H0/H1 → L1H5/H6) scored **0.000** induction fraction across all methods.

**Why**: The 5–95% firing rate filter excluded induction-specific features. The induction circuit activates only on rare repeated-token patterns (A...B...A), so its features have low firing rates and were filtered out.

**What was recovered**: All methods consistently identify **L0H9 → L1H2 / L1H10** — a broad semantic circuit driven by generally-active features.

---

## Why Not All SAE Features?

| Features (p) | Filter | PC-FisherZ | GES-BIC |
|---|---|---|---|
| 76 (current) | 5–95% | 64s | 36s |
| ~300 | 1–99% | ~8 hours | ~15 min |
| ~800 | 0.5–99.5% | ~100 days | ~5 hours |
| 49,152 (all) | none | centuries | weeks |

PC scales as O(p^5) at depth 3. GES scales as O(p^3). Running all features is infeasible.

**The right fix**: targeted feature selection — pick features attributed to specific circuit heads (e.g. L0H0/H1, L1H5/H6), keep p ≈ 80, re-run. This would directly target induction-relevant features without exploding runtime.

---

## Limitations

- PC runs skeleton phase only — orientation uses architectural prior, not full Meek's rules
- Only 40 edges validated → wide confidence intervals (~±20pp)
- Mean Ablation is not Marks et al.'s gradient-based attribution
- Ablation IA is biased toward ablation-based discovery
- 5–95% filter excludes rare circuit-specific features (induction)
- GES-BDeu not implemented (causal-learn too slow; no fast custom version)

---

## Repository Structure

```
pgm-final/
├── phase3_structure_learning.ipynb   # Main notebook: all methods, validation, metrics
├── head_level_analysis.ipynb         # Head-level circuit recovery analysis
├── make_ppt.py                       # Generates the presentation (python-pptx)
├── data/pgm-final/
│   ├── activations/
│   │   ├── X_continuous.npy          # (5000, 76) continuous SAE activations
│   │   ├── X_binary.npy              # (5000, 76) binarised activations
│   │   ├── indices_L1.npy            # (30,) SAE feature indices for L1
│   │   ├── indices_L2.npy            # (46,) SAE feature indices for L2
│   │   └── metadata.json
│   ├── data/
│   │   └── prompts.npy               # (5000, 40) token prompt sequences
│   └── results/
│       ├── attribution_matrix.npy    # (30, 46) mean-ablation sensitivity scores
│       ├── pc_fisherz_results.pkl    # PC-FisherZ edges at α={0.01,0.05,0.10}
│       ├── pc_gsq_results.pkl        # PC-GSq edges at α={0.01,0.05,0.10}
│       ├── ges_results.pkl           # GES-BIC edges
│       ├── patch_results.pkl         # Ablation validation (40 edges)
│       ├── summary_table.csv         # Full results table
│       └── method_comparison.png     # Main comparison figure
├── .gitignore
└── README.md
```

---

## Setup

```bash
# Clone and create environment
git clone <repo>
cd pgm-final
python -m venv .venv
source .venv/bin/activate
pip install transformer_lens sae_lens networkx scipy pandas tqdm matplotlib python-pptx

# Download data (not in repo — large files)
# Place activations in data/pgm-final/activations/
# Place prompts in data/pgm-final/data/

# Run main notebook
jupyter notebook phase3_structure_learning.ipynb
```

All heavy results (PC, GES, attribution matrix, patch results) are cached to `results/` after first run — subsequent runs load from cache.

---

## References

- Olsson et al. (2022). *In-context Learning and Induction Heads*. Anthropic.
- Marks et al. (2025). *Sparse Feature Circuits*. Anthropic.
- Spirtes, Glymour & Scheines (2000). *Causation, Prediction, and Search*. MIT Press.
- Chickering (2002). *Optimal Structure Identification With Greedy Search*. JMLR.
- Elhage et al. (2021). *A Mathematical Framework for Transformer Circuits*. Anthropic.
