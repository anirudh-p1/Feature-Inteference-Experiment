# Attention Kernels and Feature Interference Under a Representational Bottleneck

## Problem

Neural networks are increasingly used to make or support consequential decisions, but we cannot reliably tell what their internal representations contain, or how that information is combined. A large soure of this opacity is superposition, where networks are asked to represent more distinct features than they have available dimensions. They respond by letting unrelated features share the same representational directions rather than sacrificing capacity. When features that have nothing to do with each other end up sharing a direction, a single internal unit can respond to multiple, disconnected concepts at once, known as polysemanticity, making it hard to isolate what any part of the network is doing, and making behavior harder to predict when inputs shift away from what the model was trained on.

The typical and dominant response to this problem is post-hoc: train a model normally, then try to decompose its resulting activations after the fact (eg Sparse Autoencoders). 

This experiment asks the question: Rather than untangling representations after they've already formed, can the training process be shaped architecturally, so harmful entanglement is less likely to occur in the first place?

## Overview of the Experiment

The goal of this experiment is to essentially measure something usually invisible (whether two specific and known concepts get entangled within a trained network). Due to this, this project does not use real-world data, and uses synthetic data with a fully known ground-truth. This is a small dataset consisting of 20 binary features, where the relationships between features are defined by us. This is the same design principle that the paper 'Toy Models of Superposition, 2022' uses and is the foundation for this experiment. This does mean the experiment trades real-world realism for the ability to know, with certainity whether a model actually recovered the truth, rather than inferring it.

**The Model**
For every condition tested, the mdoel does the same four things:
1. Each of the 20 feature slots gets its own learned embedding; an active feature contributes its embedding, an inactive one contributes zero.
2. A single self-attention layer processes these 20 slot-representations — (this is the only part of the model that changes between conditions).
3. The 20 updated representations are summed into one vector (the bottleneck) of size `hidden_dim`. Compressing 20 features into fewer dimensions is what creates pressure for the network to reuse representational space in the first place.
4. A linear layer + sigmoid reconstructs the original 20 features from the bottleneck.

The model is trained to do exactly one thing: minimize reconstruction error. It is never told to be interpretable or to separate features therefore, any structure found afterward is an emergent side-effect of this objective under this architecture, not something imposed directly.

## The 3 Kernels Tested

### 1. Softmax (baseline)
$$
\text{weight}_{ij} = \frac{\exp(\text{score}_{ij})}{\sum_{k} \exp(\text{score}_{ik})}
$$

This is what real transformers use. We use this as a baseline since the ratio between the strongest and weakest attended position under softmax is unbounded so a small score difference can become an enormous weight difference. This gives a network maximum flexibility to develop very sharp, winner-take-all attention patterns, which is one plausible route by which entangled, hard-to-audit representations could form during training. Softmax is also translation-invariant so shifting every score by the same constant changes nothing.

### 2. Bounded sigmoid + floor (intervention)
$$
\text{weight}_{ij} = \frac{\text{sigmoid}(\text{score}_{ij}) + 0.05}{\sum_{k} \left(\text{sigmoid}(\text{score}_{ik}) + 0.05\right)}
$$

The small additive floor guarantees every raw weight stays above a fixed positive value, which caps the maximum-to-minimum weight ratio at roughly 21, no matter how extreme the underlying similarity scores become. If attention concentration is part of what allows harmful feature interference to develop, capping that concentration should show up as measurably less interference between features that have nothing to do with each other (that is the hypothesis this kernel exists to test). Note this kernel is not translation-invariant like softmax, and it pulls toward globally uniform attention as scores go very negative so there are real differences from softmax beyond "boundedness" alone.

### 3. Squared-ReLU hard-sparse (contrast)
$$
\text{weight}_{ij} = \frac{\text{relu}(\text{score}_{ij})^2}{\sum_{k} \text{relu}(\text{score}_{ik})^2}
$$

Unlike the smooth cap above, this gives any position with a negative similarity score exactly zero weight — a hard cutoff rather than an asymptotic approach to zero. This tests a fundamentally different, harder kind of constraint: if the bounded kernel's smooth cap helps, does an even more aggressive hard cutoff help more, or is it too aggressive and destroys useful weak signal? This is not the sparsemax operator, it's a simpler hard-sparse approximation. Because it behaves so differently from the other two (see Results), it is kept as a separate contrast condition, not folded into the formal statistical comparison between softmax and bounded.

## The Data
Two datasets, both 20 binary features, generated in code.

- **Dataset A (Independent):** all 20 features activate independently, each with p=0.1.
- **Dataset B (Structured):** features 0–7 independent (control group); features 8–13 form three **positively correlated** pairs (if A is on, B is on 80% of the time; if A is off, B is on only 5% of the time); features 14–19 form three mutually exclusive pairs (if A is on, B is never on).

This design lets a single set of training runs answer three questions at once: does a kernel change how the network treats features that are (1) genuinely unrelated, (2) correlated in a way that could be usefully exploited, or (3) mutually exclusive, where sharing space costs nothing regardless of kernel.

**Empirical validation** (measured on 100,000 generated samples, not assumed): positive pairs show Pearson r ≈ 0.72; exclusive pairs show r ≈ −0.16 with a measured co-activation rate of exactly 0.0 — confirming the generator produces the intended relationships.

## The 2×2 Factorial Design

Rather than testing kernels at only one bottleneck size, it crosses kernel (softmax vs. bounded) with bottleneck capacity (`tight`: hidden_dim=8, vs. `wide`: hidden_dim=20), asking a sharper question than "does the kernel matter" alone: **does the kernel's effect depend on how much compression pressure the network is under?**

- If a kernel effect only appears at the tight bottleneck, that points to the effect being specifically about compression pressure.
- If it appears at both, that suggests something more general.
- If it disappears entirely at the wide bottleneck, that suggests the bottleneck — not the kernel — is the real driver of any interference observed.

The sparse kernel is run at both bottleneck widths too, but as a separate contrast, not part of this formal comparison.

**Full run count:** 5 random seeds × 2 kernels × 2 bottleneck widths × 2 datasets = 40 fits for the formal factorial, plus 5 seeds × 1 kernel × 2 widths × 2 datasets = 20 fits for the sparse contrast = **60 total model fits.**

## What we measured

- **Reconstruction loss (BCE) and balanced accuracy** — the guardrail metric. A kernel that appears to reduce interference while destroying the network's ability to do its actual job isn't a success by any reasonable definition.
- **Ridge-based feature-direction interference** — fit a ridge regression estimating each feature's linear direction in the bottleneck (ridge rather than plain least squares, because correlated features make the regression poorly conditioned), then measure pairwise cosine similarity between these estimated directions, reported separately for independent, positively-correlated, and exclusive feature groups. This is the primary geometric measurement of whether features overlap.
- **Intervention-based feature-direction interference** — a second, more directly causal version of the same idea: for real examples, force one feature on vs. off while holding everything else fixed, and measure how much the bottleneck representation actually shifts. This can't be fooled by two features merely co-occurring in the data without the network actually treating them similarly, which the regression-based metric alone can be.
- **Linear probe recovery (AUROC, average precision, balanced accuracy)** — for each of the 20 known ground-truth features, train a simple probe to recover it from the frozen bottleneck, using data pools kept completely separate from the network's own training and test data (to avoid leakage). AUROC and balanced accuracy are used instead of raw accuracy because, at ~10% feature activation frequency, a probe that always predicts "off" would score ~90% accuracy while being useless.
- **Random-direction null baseline** — 1,000 sets of 20 *completely random, model-unrelated* directions are generated at each bottleneck width, and the same interference metric is computed on them. This measures how much apparent "interference" exists purely from being in a low-dimensional space, independent of anything the network actually learned. Any measured interference at or below this level is not evidence of real entanglement.
- **Mechanistic attention statistics** (entropy, max attention weight, effective support size, fraction of exact-zero weights) — measured directly rather than assumed from each kernel's formula, to confirm the kernels actually behave the way their equations suggest (e.g., is squared-ReLU actually sparse in practice, or not).

## Results
**1. Independent features: little real interference for softmax or bounded to reduce.** At the tight bottleneck, both softmax and bounded land at mean|cos| ≈ 0.25 on independent (Dataset A) features — *below* the random-null baseline of ≈0.29 at that same width. There was not much excess interference above chance for either kernel to act on in the first place. This is directly supported by the mechanistic statistics: softmax's own attention entropy sits close to the theoretical maximum (ln 20 ≈ 2.996) across most conditions, and its max attention weight stays small (~0.05–0.16) — it was never behaving in the sharp, winner-take-all way the original hypothesis assumed it might.

**2. Squared-ReLU (hard-sparse) consistently underperforms, disconfirming the "harder cutoff helps more" hypothesis.** It shows higher independent-feature interference than the other two kernels (often at or above the random-null baseline), worse reconstruction loss, and — most notably — meaningfully worse exploitation of positive-correlated structure (~0.61–0.68 mean|cos| vs. ~0.68–0.85 for softmax/bounded). Forcing hard cutoffs cost real performance without buying any interference reduction.

**3. Positive-correlated features show strong, clearly above-chance shared structure, across all three kernels.** Cosine similarities of ~0.68–0.85 sit far above the ~0.18–0.29 random-chance range measured at each bottleneck width — a clean confirmation that the network exploits genuine statistical structure between correlated features rather than merely suffering it as noise.

**4. The kernel × bottleneck interaction is small, within the precision available.** The bounded-minus-softmax differences (e.g. −0.012 at tight/B_structured, +0.006 at wide/B_structured) are comparable in magnitude to the seed-to-seed standard deviations observed elsewhere in the data (~0.02–0.05). At 5 seeds, this is not distinguishable from noise — the honest claim is that kernel choice's effect on independent-feature interference did not detectably depend on bottleneck width at this sample size, not that no such dependence exists at all.

**5. Exclusive (mutually-exclusive) pairs are the least conclusive group.** Their interference values are in a broadly similar range to positive pairs but with much wider run-to-run variability (partly because only 3 pairs contribute to each average). This is flagged as suggestive rather than conclusive, and as a natural target for more seeds or more pairs in future work.

## Reproducibility

All randomness (data generation, model initialization, training) is explicitly seeded (`seeds = 0–4`). Re-running `experiment.py` reproduces identical results.
