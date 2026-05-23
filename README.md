# Fasciculation Detection Using Ultrasound Data

> Bayesian Black-Box Variational Inference (BBVI) with EM optimization and multi-scale image pyramids for detecting and estimating fasciculation-like rotational motion in ultrasound video frames.

---

## Table of Contents

- [Project Description](#project-description)
- [Objectives](#objectives)
- [Repository Structure](#repository-structure)
- [Dependencies](#dependencies)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration Parameters](#configuration-parameters)
- [Algorithm Overview](#algorithm-overview)
- [Inputs](#inputs)
- [Outputs](#outputs)
- [Known Limitations](#known-limitations)
- [Related Repository](#related-repository)
- [License](#license)

---

## Project Description

This project implements a probabilistic rotation estimation pipeline for detecting fasciculation-like motion in medical video data (e.g., ALS muscle ultrasound). The algorithm models each annular ring region around a center point as having an independent rotation angle, and estimates these angles frame-by-frame using:

- **Black-Box Variational Inference (BBVI)** with Monte Carlo gradient estimation (ELBO maximization)
- **Expectation-Maximization (EM)** with separate E-step (rotation angles) and M-step (center + noise)
- **Multi-scale image pyramids** (coarse-to-fine: scales 0.25 → 0.5 → 1.0)
- **Bayesian prior fusion** via product-of-Gaussians, combining the previous frame's posterior with the coarser pyramid level's posterior as the prior for each finer level

The ground truth initial rotation angles are loaded from a CSV file produced by the companion simulation repository (`Fasciculation-Simulation`).

---

## Objectives

1. **Estimate per-ring rotation angles** from consecutive video frame pairs using variational inference.
2. **Track the rotation center** dynamically across frames using gradient-based M-step optimization.
3. **Fuse temporal and spatial uncertainty** via Bayesian product-of-Gaussians priors across pyramid levels and frame pairs.
4. **Quantify posterior uncertainty** — track per-ring posterior variance over time as a diagnostic.
5. **Visualize results** — rotation trajectories, convergence metrics (MSE, ELBO, noise variance), center trajectory, and posterior variance plots.

---

## Repository Structure

```
fasciculation-detection/
│
├── src/
│   └── fasciculation_detection_bbvi.py    # Main BBVI + EM detection script
│
├── data/
│   └── Original_Groundtruth_Values.csv    # Initial rotation angles CSV (user-provided)
│
├── outputs/                               # Plots and result arrays (gitignored)
│   └── README.md
│
├── docs/
│   └── algorithm_overview.md              # Detailed algorithm documentation
│
├── assets/                                # Sample output plots or preview images
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Dependencies

| Package          | Version (recommended) | Purpose                                        |
|------------------|-----------------------|------------------------------------------------|
| Python           | ≥ 3.8                 | Runtime                                        |
| numpy            | ≥ 1.21                | Array operations                               |
| opencv-python    | ≥ 4.5                 | Video loading, image resizing                  |
| pandas           | ≥ 1.3                 | CSV loading for initial angles                 |
| matplotlib       | ≥ 3.4                 | Result visualization                           |
| torch (PyTorch)  | ≥ 1.11                | Tensor ops, autograd, grid_sample              |
| tqdm             | ≥ 4.62                | Progress bars                                  |

GPU support is automatic — the script uses CUDA if available, otherwise falls back to CPU.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/ShzdiTrnum/Fasciculation-Detection-Using-Synthetic-Data.git
cd Fasciculation-Detection-Using-Synthetic-Data
```

### 2. Create a virtual environment

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

For GPU support, install the appropriate PyTorch version from https://pytorch.org/get-started/locally/

---

## Usage

### Running on Google Colab (recommended)

Mount your Google Drive and update the paths at the top of the script:

```python
video_path      = "/content/drive/MyDrive/YOUR_FOLDER/your_video.mp4"
csv_init_angles = "/content/drive/MyDrive/YOUR_FOLDER/Original_Groundtruth_Values.csv"
```

Then run:

```bash
python src/fasciculation_detection_bbvi.py
```

### Running locally

Update the same two paths to point to your local files, then run the script directly.

---

## Configuration Parameters

All key parameters are defined at the top of the script:

| Parameter          | Default          | Description                                                          |
|--------------------|------------------|----------------------------------------------------------------------|
| `video_path`       | —                | Path to input `.mp4` video file                                      |
| `csv_init_angles`  | —                | Path to CSV with initial ground-truth rotation angles per ring       |
| `FRAME_START`      | 6                | First frame to process (1-based)                                     |
| `FRAME_END`        | 24               | Last frame to process (1-based, inclusive)                           |
| `xmin/xmax`        | 250, 500         | ROI x-bounds in pixels                                               |
| `ymin/ymax`        | 200, 450         | ROI y-bounds in pixels                                               |
| `radii`            | `arange(5,101,5)`| Ring radii in pixels — 20 rings at 5px intervals up to 100px        |
| `pyramid_scales`   | [0.25, 0.5, 1.0] | Multi-scale pyramid levels (coarse to fine)                          |
| `S_mc`             | 30               | Number of Monte Carlo samples per ELBO estimate                      |
| `OUTER_EM_ITERS`   | 60               | Number of outer EM iterations per pyramid level                      |
| `E_UPDATES`        | 30               | Inner E-step gradient steps per outer iteration                      |
| `M_UPDATES`        | 10               | Inner M-step gradient steps per outer iteration                      |
| `Elr_levels`       | [1e-4, 1e-4, 1e-4] | E-step Adam learning rates per pyramid level                      |
| `Mlr_levels`       | [1e-4, 1e-4, 1e-4] | M-step Adam learning rates per pyramid level                      |

---

## Algorithm Overview

### 1. Frame Loading & ROI Cropping
Video frames are loaded with OpenCV and cropped to the user-defined ROI.

### 2. Radial Ring Masks
20 concentric annular masks are computed at 5-pixel radial intervals from the ROI center.

### 3. Multi-scale Image Pyramid
Each frame pair is downsampled to 3 scales. Processing proceeds coarse → fine.

### 4. Variational Inference (E-step)
For each pyramid level, the posterior `q(θ) = N(μ, σ²)` is updated by maximizing the ELBO using Monte Carlo gradient estimates. Each ring has an independent `(μ, logσ²)` pair.

### 5. Center & Noise Estimation (M-step)
The rotation center and observation noise `σ_n` are updated by minimizing mean squared warping error. After the first frame pair, `σ_n` is fixed.

### 6. Bayesian Prior Fusion
When propagating from a coarse pyramid level to the next finer level, the prior is formed by the product of two Gaussians:
- The posterior from the current coarser level
- The posterior from the previous frame (already seeded into the next-level parameters)

This is standard Bayesian precision addition: `prec_fused = prec_coarse + prec_prev_frame`.

### 7. Temporal Propagation
The finest-level posterior `(μ, logσ²)` from each frame pair seeds all pyramid levels for the next pair.

---

## Inputs

| Input | Format | Description |
|-------|--------|-------------|
| Video file | `.mp4` | Grayscale or BGR ultrasound/synthetic video |
| Initial angles CSV | `.csv` | Columns: `Frame`, `Rectangle_0` … `Rectangle_19` — initial rotation angles in radians for 20 rings |

The initial angles CSV is produced by the `Fasciculation-Simulation` repository.

---

## Outputs

The script produces the following visualizations (displayed inline or saved):

| Output | Description |
|--------|-------------|
| Rotation angle plot | Per-ring rotation angle (radians) vs. frame index |
| Noise variance plot | `σ_n` convergence across EM iterations per pyramid level |
| MSE plot | Mean squared warping error per pyramid level |
| ELBO plot | Evidence lower bound per pyramid level |
| Center trajectory | 2D scatter plot of rotation center drift over frames |
| Posterior variance plot | Per-ring posterior variance over time (uncertainty diagnostic) |

---

## Known Limitations

- Input paths are hardcoded for Google Colab/Drive. Update before local use.
- No CLI argument parser — all config is edited directly in the script.
- Convergence is not verified automatically; monitor ELBO/MSE plots manually.
- The model assumes rotationally symmetric motion within each annular ring.
- Performance on CPU is significantly slower than on GPU; a GPU is strongly recommended.
  
---

## Related Repository

This repository uses initial rotation angle values generated by:

**[Fasciculation-Simulation](https://github.com/ShzdiTrnum/Fasciculation-Simulation)**
> Synthetic video simulation of concentric ring rotation with acceleration/deceleration dynamics.

---

## License

This project is released under the MIT License. See `LICENSE` for details.
