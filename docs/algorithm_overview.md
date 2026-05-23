# Algorithm Overview

This document provides a detailed explanation of the BBVI + EM rotation estimation algorithm used in `fasciculation_detection_bbvi.py`.

---

## Problem Statement

Given two consecutive grayscale video frames `f1` and `f2` (cropped to a Region of Interest), estimate the rotation angle `θ_r` for each of `R` concentric annular rings centered at a point `(cx, cy)`. The rotation model assumes that the pixel intensity pattern in ring `r` of `f1` can be recovered by rotating the corresponding region of `f2` by angle `θ_r` around the center.

---

## Probabilistic Model

### Observation model

```
f1(x) ≈ f2(R(θ_r, c) · x) + ε,    ε ~ N(0, σ_n²)
```

Where `R(θ_r, c)` is a rotation transformation by `θ_r` radians around center `c`.

### Latent variable

```
θ = {θ_r : r = 1 ... R}    (R = 20 independent rotation angles)
```

### Prior (fused Gaussian)

At each pyramid level and frame pair, a Gaussian prior is formed by the product of:
1. The posterior from the previous frame
2. The posterior from the coarser pyramid level

```
p(θ) ∝ N(θ | μ_prev, σ²_prev) · N(θ | μ_coarse, σ²_coarse)
     = N(θ | μ_fused, σ²_fused)
```

Where precision adds:
```
1/σ²_fused = 1/σ²_prev + 1/σ²_coarse
μ_fused    = σ²_fused · (μ_prev/σ²_prev + μ_coarse/σ²_coarse)
```

---

## Variational Inference (E-step)

The variational posterior is mean-field Gaussian:

```
q(θ) = ∏_r N(θ_r | μ_r, σ²_r)
```

Parameters `{μ_r, log σ²_r}` are optimized by maximizing the Evidence Lower Bound (ELBO):

```
ELBO = E_q[log p(f1 | θ, c, σ_n)] - KL[q(θ) || p(θ)]
```

The expectation is approximated with `S_mc = 30` Monte Carlo samples using the reparameterization trick:

```
θ^(s) = μ + σ · ε^(s),    ε^(s) ~ N(0, I)
```

The log-likelihood term is:

```
log p(f1 | θ, c, σ_n) = -1/(2σ_n²) · ||f1 - warp(f2, θ, c)||²_F
```

---

## Center & Noise Estimation (M-step)

The rotation center `c = (cx, cy)` and noise variance `σ_n` are updated by minimizing the expected squared warping error:

```
min_{c, σ_n}  E_q[||f1 - warp(f2, θ, c)||²]
```

`σ_n` is additionally updated analytically:

```
σ_n² = E_q[||f1 - warp(f2, θ, c)||²] / (H · W)
```

After the first frame pair, `σ_n` is frozen at its estimated value.

---

## Multi-Scale Pyramid

Processing proceeds from coarsest (scale = 0.25) to finest (scale = 1.0). At each level transition:

1. The coarse-level posterior `q_coarse(θ)` is computed.
2. The previous-frame posterior `q_prev(θ)` (seeded into the next level) acts as temporal prior.
3. A Bayesian product-of-Gaussians fusion produces the initialisation for the next finer level.
4. The rotation center is scaled proportionally.

This coarse-to-fine strategy prevents the optimization from getting stuck in local minima due to large displacements.

---

## Temporal Propagation

For frame pair `t > 0`, the finest-level posterior `(μ_fin, logσ²_fin)` from pair `t-1` is used to seed all pyramid levels of pair `t`. This enables temporal continuity in both the mean estimates and the uncertainty.

---

## Warping Implementation

Rotation warping uses `torch.nn.functional.grid_sample` with bilinear interpolation. The sampling grid is computed analytically from the rotation angle map:

```
x_warped = cos(θ) · (x - cx) - sin(θ) · (y - cy) + cx
y_warped = sin(θ) · (x - cx) + cos(θ) · (y - cy) + cy
```

The grid is normalized to `[-1, 1]` as required by `grid_sample`.

---

## Output Diagnostics

| Plot | What to look for |
|------|-----------------|
| Rotation angle vs frame | Smooth temporal trajectories indicate good convergence |
| ELBO vs iteration | Should increase monotonically within each level |
| MSE vs iteration | Should decrease; plateau indicates convergence |
| Noise variance | Should stabilize after first few outer iterations |
| Posterior variance | Should decrease or stay low — increasing variance suggests poor fitting |
| Center trajectory | Should remain within or near the ROI boundary |
