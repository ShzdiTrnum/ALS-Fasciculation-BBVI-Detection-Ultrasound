# Algorithm overview and implementation notes

This page describes what the code in `src/` **actually computes**, step by step,
and lists where the implementation is simpler than the full formulation in the paper.
Reading it next to the source should take about ten minutes.

---

## 1. Inputs

* Two grayscale frames `f1 = frame[i]` and `f2 = frame[i + FRAME_STEP]`, cropped to the ROI and scaled to `[0, 1]`.
* Concentric annular masks around the **ROI centre**: ring *k* contains the pixels whose distance
  from the ROI centre lies in `(r_k − Δ/2, r_k + Δ/2]`, with `r_k` from `--radii` and `Δ = radii[0]`
  (`make_radial_masks`). Pixels outside every ring get θ = 0 in the warp.
  - Ultrasound defaults (5, 10, …, 100): `Δ = 5` = ring spacing, so the rings tile the disc.
  - Synthetic defaults (10, 30, …, 190): `Δ = 10` but the spacing is 20, so each ring is 10 px wide and the
    10 px gaps between rings (e.g. 15–25 px) are warped with θ = 0.
* An initial rotation angle per ring, read from the **first data row** of `--init-csv`.

## 2. Observation model

```
f1(x) = f2( R(θ_k) (x − c) + c ) + n,     n ~ N(0, σ_n²),   x in ring k
R(θ)  = [[cos θ, −sin θ], [sin θ, cos θ]]
```

The warp is done with `torch.nn.functional.grid_sample` (bilinear, border padding).
The rotation centre `c` is a free parameter; the ring masks themselves stay fixed around the ROI centre.

**Sign convention.** With this model, content that rotates counter-clockwise on screen
(OpenCV's positive angle) from `f1` to `f2` gives a **negative** `θ`.

## 3. Variational posterior

Mean-field Gaussian with one independent angle per ring (the "unstructured" parameterisation):

```
q(θ) = Π_k N(θ_k | μ_k, σ_k²)      parameters: μ_k, log σ_k²
θ^(s) = μ + σ ⊙ ε^(s),  ε^(s) ~ N(0, I)      (reparameterisation trick)
```

## 4. E-step (as implemented)

The E-step minimises the Monte-Carlo estimate of the expected photometric error, scaled by the noise level:

```
loss_E = (1 / 2σ_n²) · (1/S) Σ_s || f1 − warp(f2, θ^(s), c) ||²
```

with Adam on `(μ, log σ²)`. On the **first frame pair** the gradient of `μ` is zeroed, so `μ` stays at the
values from `--init-csv` and only `σ` is updated.

> **Note.** In this version the KL term is set to zero (`kl = 0` in the ultrasound script; absent in the
> synthetic script), and there is no entropy term. The quantity plotted as "ELBO (Log-Likelihood)" is the
> expected log-likelihood term only. Without the KL/entropy terms nothing prevents `σ_k` from shrinking,
> so the posterior variances should be read as a diagnostic, not as calibrated uncertainty.

## 5. M-step (as implemented)

* `c` is updated with Adam on the same expected squared error.
* After each outer iteration, `σ_n² = (1/S) Σ_s ||f1 − warp(f2, θ^(s), c)||² / (H·W)` (closed form).
* The scripts store the first pair's `σ_n` per level and set it at the start of later pairs, but the closed-form
  update above then overwrites it on every outer iteration, so in practice `σ_n` keeps being re-estimated.

## 6. Coarse-to-fine pyramid and prior fusion

Scales `0.25 → 0.5 → 1.0`. After a level converges, its posterior is combined with the posterior that was
seeded into the next level (the previous frame pair's result) by a product of Gaussians:

```
1/σ²_fused = 1/σ²_coarse + 1/σ²_prev
μ_fused    = σ²_fused · (μ_coarse/σ²_coarse + μ_prev/σ²_prev)
```

`(μ_fused, log σ²_fused)` is used as the **starting point** of the next level's optimisation
(it does not enter the loss as a prior, because the KL term is zero). The centre is rescaled to the next level.

## 7. Temporal propagation

The finest-level `(μ, log σ²)` of pair *t* seeds every level of pair *t + 1*. The centre is reset to the ROI
centre at the start of every pair.

## 8. Differences from the paper's full formulation

| Paper | This code |
|---|---|
| ELBO = E_q[log p] − KL(q‖p) (Eq. 7, 10) | KL term omitted (see §4) |
| Fused Gaussian used as the prior p(w) (Eq. 20 in the journal draft) | Fused Gaussian used as the initialisation (§6) |
| Structured basis-function posterior (Sec. 4.2) | Not in this repository — independent per-ring angles only |
| Log-normal reparameterisation (VISAPP Sec. 4.3) | Not implemented |
| Green's-theorem initialisation | Initial angles read from a CSV produced by the `Optical-Flow` repo (ultrasound) or the simulation (synthetic) |
| Radius `r` measured from the estimated centre `c` (Eq. 2) | Ring masks fixed around the ROI centre |

## 9. Diagnostics produced

| File | What it shows |
|---|---|
| `rotation_per_radius.*` | posterior mean `μ_k` per ring vs. source frame |
| `posterior_variance.*` | `σ_k²` per ring vs. source frame (see note in §4) |
| `convergence_mse.*` | mean photometric error per outer EM iteration, each level (last frame pair) |
| `convergence_noise_variance.*` | `σ_n` per outer iteration, each level (last frame pair) |
| `convergence_elbo.*` | expected log-likelihood term per outer iteration, each level (last frame pair) |
| `center_trajectory_roi.*` / `center_trajectory_image_coords.*` | estimated centre per pair, ROI or full-image pixel coordinates |
