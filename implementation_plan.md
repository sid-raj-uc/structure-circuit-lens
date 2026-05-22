# Implementation Plan: Graphical Model Structure Learning for Sparse Feature Circuits

**Course:** TTIC 31180 — Final Project  
**Deadline:** ~May 28, 2026 (5 weeks from April 23)  
**Status as of May 8:** ~2.5 weeks in, ~2.5 weeks remaining

---

## Project Goal

Replace attribution patching (gradient-based, conflates direct/indirect effects) with classical PGM structure learning to recover causally faithful feature circuits in GPT-2 small's induction circuit. Compare PC algorithm and GES against the attribution patching baseline using direct activation patching as causal ground truth.

---

## Repository Structure

```
pgm-final/
├── data/
│   ├── generate_prompts.py        # induction-style prompts
│   └── collect_activations.py     # GPT2 + SAE hooks → .npy files
├── methods/
│   ├── attribution_patching.py    # thin wrapper over saprmarks/feature-circuits
│   ├── pc_algorithm.py            # causal-learn PC with fisherz and gsq CI tests
│   └── ges.py                     # causal-learn GES with BIC and BDeu scores
├── evaluation/
│   ├── activation_patching.py     # direct causal ground truth (zero-ablation)
│   └── metrics.py                 # intervention agreement, precision/recall
├── notebooks/
│   └── analysis.ipynb             # comparison plots, graph visualization
└── requirements.txt
```

---

## Dependencies

```
torch
transformer_lens
sae_lens          # pretrained SAEs for GPT-2 small (Joseph Bloom's)
causal-learn      # PC algorithm + GES built-in
numpy
pandas
matplotlib
networkx          # graph visualization
scipy
```

Use `sae_lens` — do not hand-roll SAE loading. Use `causal-learn` for PC and GES — do not reimplement from scratch.

---

## Phase 1 — Environment + Data Pipeline

**Target: Done by May 10**

### 1a. Generate induction prompts (`data/generate_prompts.py`)

Induction circuit task: `[A][B]...[A] → B` (random token pair, repeated in same context).

- Generate **5,000 prompts** (need N >> p for CI tests to have power)
- Save as list of token sequences

### 1b. Collect SAE feature activations (`data/collect_activations.py`)

- Load GPT-2 small via `TransformerLens`
- Load pretrained SAEs from `sae_lens` at **layers 0 and 1 residual stream** — that's where the induction circuit lives (prev-token head L0, induction head L1)
- Hook SAE encoder outputs; save activation matrix: shape `(N_prompts, N_features)`
- Save to `.npy` files on disk — everything downstream loads from these

### 1c. Feature filtering

Filter to **top 100–200 task-relevant features per layer** by mean activation frequency (fire rate > 5% of prompts). Do not attempt structure learning over all 10,000+ SAE features — CI tests will be underpowered and computation intractable.

**Fallback:** If SAE integration takes more than 3 days, switch to attention-head level activations (hook attention pattern outputs directly from TransformerLens — no SAEs needed). This is 1 day of work and still tests the core scientific question.

---

## Phase 2 — Structure Learning Methods

**Target: Done by May 20**

### 2a. Attribution Patching Baseline (`methods/attribution_patching.py`)

Clone `saprmarks/feature-circuits` (Marks et al. 2024 public code). Write a thin wrapper that runs their edge attribution patching on the same prompt set and returns a scored edge list. This is both the baseline to beat and a rough pre-filter for which edges are plausibly non-zero.

### 2b. PC Algorithm (`methods/pc_algorithm.py`)

```python
from causallearn.search.ConstraintBased.PC import pc

# Pass 1: Gaussian approximation (fast, wrong in theory, good sanity check)
cg_fisherz = pc(data, alpha=0.05, indep_test='fisherz')

# Pass 2: Binary CI test (honest about sparse zero-inflated data)
# Threshold features: firing = 1, silent = 0
cg_gsq = pc(data_binary, alpha=0.05, indep_test='gsq')
```

Run both. If they agree, report convergence. If they disagree, that disagreement is itself a result worth reporting in the paper.

Sweep `alpha` in {0.01, 0.05, 0.10} and report sensitivity — guards against over/underfitting the graph.

### 2c. GES (`methods/ges.py`)

```python
from causallearn.search.ScoreBased.GES import ges

# Pass 1: BIC for continuous data
Record_bic = ges(data, score_func='local_score_BIC')

# Pass 2: BDeu for binary/discrete data
Record_bdeu = ges(data_binary, score_func='local_score_BDeu')
```

### 2d. NOTEARS (stretch goal only)

Use `dagma` package. Only implement if Phases 2a–2c are complete and time remains. Not required for the core story.

---

## Phase 3 — Evaluation + Validation

**Target: Done by May 25**

### 3a. Causal ground truth (`evaluation/activation_patching.py`)

For a **sample of ~30–50 candidate edges** where PC/GES and attribution patching disagree:
- Zero-ablate the source SAE feature (set activation to 0)
- Measure change in downstream feature activation
- Record as confirmed/disconfirmed

This is expensive per-edge — pick the most informative disagreements, not all edges.

### 3b. Primary metric: Intervention Agreement (`evaluation/metrics.py`)

For each method (PC-fisherz, PC-gsq, GES-BIC, GES-BDeu, attribution patching):
- Among edges it predicts, what fraction is confirmed by direct activation patching?

This directly answers the core question: do PGM methods produce more causally faithful edges than attribution patching?

### 3c. Secondary: Head-level circuit recovery

Aggregate feature-level edges to head level by summing edge weights across features belonging to the same attention head. Check if the known **prev-token-head (L0) → induction-head (L1)** connection is recovered. This is a sanity check with published ground truth (Olsson et al., 2022).

### 3d. Additional metrics

- Pairwise method agreement (edge overlap between methods)
- Graph sparsity (number of edges per method)
- Runtime comparison (attribution patching's main appeal is speed)

---

## Priority Ordering

| Priority | Task | Estimate |
|---|---|---|
| P0 | Data pipeline: prompts + activations saved to disk | 2–3 days |
| P0 | Attribution patching baseline running | 1–2 days |
| P1 | PC algorithm (fisherz + gsq, alpha sweep) | 2–3 days |
| P1 | GES (BIC + BDeu) | 1–2 days |
| P1 | Activation patching validation on ~40 disagreements | 2–3 days |
| P2 | Head-level aggregation and circuit recovery check | 1 day |
| P2 | Plots, comparison table, writeup | 3–4 days |
| P3 | NOTEARS | only if ahead of schedule |

---

## Key Design Decisions

**Variable granularity:** Individual SAE features (not grouped), filtered to top-K by activation frequency. Grouping loses the interpretability benefit of SAEs.

**CI test for sparse data:** Fisher-Z first for speed, then G-test on binarized features. Report both; the comparison is informative.

**Number of samples needed:** With 100–200 features and order-2 conditional independence tests, N=2,000 gives marginal power; N=5,000 is comfortable. Collect 5,000 prompts.

**Layers to focus on:** L0 and L1 residual streams only. This keeps the graph tractable and covers the known induction circuit.

---

## Risks and Mitigations

| Risk | Mitigation |
|---|---|
| SAE pipeline more complex than expected | Fall back to attention-head level activations (no SAEs, hook TransformerLens directly) |
| CI tests underpowered on sparse data | Use 5,000 prompts; binarize features; report sensitivity to alpha |
| PC/GES returns dense or empty graph | Sweep regularization parameters; report sensitivity as a finding |
| Too many features for structure learning | Pre-filter to top-K; 100–150 features is tractable for both PC and GES |
| Activation patching ground truth too expensive | Sample ~40 disagreement edges strategically rather than exhaustively |

---

## What a Successful Paper Looks Like

Even without NOTEARS, a clean paper showing:
1. PC/GES recover the known head-level induction circuit (sanity check)
2. PC/GES produce sparser, more causally faithful edge sets than attribution patching
3. Quantified via intervention agreement on a sample of edges

...is a genuine contribution. The framing from the proposal is correct: "even a careful empirical comparison — with intervention-based validation — would be useful."
