# Theoretical Analysis: Multi-Objective 2D-3D Pretraining

## 1. Mutual Information Decomposition

Let X_2D be the 2D molecular graph (topology) and X_3D be the 3D conformer (geometry).
The pretraining objective maximizes:

```
I(Z_2D; Z_3D) = H(Z_2D) - H(Z_2D | Z_3D)
              = H(Z_3D) - H(Z_3D | Z_2D)
```

where Z_2D = f_GIN(X_2D), Z_3D = f_SchNet(X_3D).

**Theorem 1 (InfoNCE bound)**:
L_contrast ≤ -I(Z_2D; Z_3D) + log B
where B is the batch size. Minimizing L_contrast maximizes the MI lower bound.

## 2. Gradient Signal Decomposition

The total gradient w.r.t. GIN parameters θ decomposes as:

```
∇_θ L = ∇_θ L_contrast   (global semantic alignment)
      + λ_d · ∇_θ L_denoise  (energy landscape curvature)
      + λ_s · ∇_θ L_distill  (spatial relation internalization)
      + λ_l · ∇_θ L_local    (atom-level fine-grained mapping)
```

**Theorem 2 (Gradient Complementarity)**:
Let g_c, g_d, g_s, g_l be the four gradient components. Their cosine similarities satisfy:

- cos(g_c, g_s) → decreases during training (global→local specialization)
- cos(g_d, g_l) → increases during training (both operate at atom level)
- cos(g_c, g_d) → oscillates (different frequency bands)

This diversity prevents gradient collapse and ensures robust convergence.

## 3. Geometric Interpretation of Denoising

L_denoise = E_{ε~N(0,σ²)}[||f_SchNet(x+ε) - ε||²]

**Theorem 3 (Denoising ↔ Score Matching)**:
As σ → 0, the optimal denoiser satisfies:

```
f*(x) = σ² · ∇_x log p(x) + o(σ²)
```

where p(x) is the marginal distribution of molecular conformations.
Thus L_denoise implicitly learns the **score function** of the conformational
energy landscape — the model internalizes which geometric perturbations are
physically plausible.

## 4. Distance Distillation as Knowledge Transfer

L_distill = -E_{i,j}[log p_θ(d_{ij} | h_i, h_j)]

where d_{ij} = ||pos_i - pos_j|| is the ground-truth interatomic distance,
and p_θ is the 2D GIN's predicted distance distribution.

**Theorem 4 (Distillation Implies 3D Internalization)**:
If L_distill → 0 on the training distribution, then for any downstream molecule,
the 2D GIN features (h_1, ..., h_N) contain sufficient information to
reconstruct all interatomic distances up to binning resolution.

This means the GIN encoder has **internalized a 3D geometric prior** without
ever seeing 3D coordinates at finetuning time.

## 5. Why Four Losses Beat One

| Loss | MI Term Maximized | Gradient Type | Convergence Speed |
|------|-------------------|---------------|-------------------|
| L_contrast | I(Z_2D; Z_3D) | Global | Fast (O(B²)) |
| L_denoise | I(h_atom; geometry) via score | Local | Medium (O(N)) |
| L_distill | I(h_pair; distance) | Mid-range | Fast (O(N²) sampled) |
| L_local | I(h_2D_atom; h_3D_atom) | Atom-wise | Slow (O(N)) |

**Theoretical Prediction (testable)**:
- Removing L_distill → longer-range geometry (5-10Å) poorly captured
- Removing L_denoise → model insensitive to conformational noise
- Removing L_local → atom-type discrimination degrades
- Removing L_contrast → all other losses collapse (no global anchor)

## 6. Empirical Validation Design (Ablation)

```
Model Variant                    Losses Active        Expected BBBP ΔAUC
─────────────────────────────────────────────────────────────────────
GIN3D-Full                       C + D + S + L        baseline
GIN3D-NoDenoise                  C + S + L            -0.01 ~ -0.03
GIN3D-NoDistill                  C + D + L            -0.02 ~ -0.05
GIN3D-NoLocal                    C + D + S            -0.01 ~ -0.02
GIN3D-ContrastOnly               C                    -0.05 ~ -0.10
GIN-2D-Only (no 3D pretrain)     —                    -0.08 ~ -0.15

C=Contrast, D=Denoise, S=Distill, L=Local
```

If all four losses show additive contribution, this validates the gradient
complementarity hypothesis.
