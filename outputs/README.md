# outputs/

This directory stores generated plots and result arrays after running the detection script.

**Contents after a run (if saving is enabled):**

| File | Description |
|------|-------------|
| `rotation_angles.png` | Per-ring rotation angle vs. frame index |
| `noise_variance.png` | Noise variance convergence per pyramid level |
| `mse.png` | MSE per pyramid level |
| `elbo.png` | ELBO per pyramid level |
| `center_trajectory.png` | 2D center drift plot |
| `posterior_variance.png` | Per-ring uncertainty over time |

> **Note:** Output files are excluded from version control via `.gitignore`.
> Add them manually or upload as GitHub Release assets if you wish to share results.
