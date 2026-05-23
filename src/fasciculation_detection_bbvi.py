import os
import math
import copy
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from tqdm import trange

plt.rcParams.update({
    'font.size': 18,
    'axes.titlesize': 18,
    'axes.labelsize': 18,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'legend.fontsize': 14,
    'figure.titlesize': 16
})

import torch
import torch.nn.functional as F
from torch import nn

# -----------------------------
# User-specified paths & params
# -----------------------------
video_path = "/content/drive/MyDrive/VI/ALS - Trim20s.mp4"
csv_init_angles = "/content/drive/MyDrive/VI/Original_Groundtruth_Values.csv"
FRAME_START = 6
FRAME_END = 24
# ROI
xmin, xmax = 250, 500
ymin, ymax = 200, 450
# Radii: 5-pixel intervals up to 100
radii = np.arange(5, 101, 5)   # 5..100 -> 20 rings
num_radii = len(radii)
# Pyramid scales (coarse -> fine)
pyramid_scales = [0.25, 0.5, 1.0]  # Level 0: coarsest, Level 2: finest
num_levels = len(pyramid_scales)
# BBVI / EM hyperparams
S_mc = 30           # MC samples
OUTER_EM_ITERS = 60
E_UPDATES = 30
M_UPDATES = 10

# Learning rates per level (coarser -> finer)
Elr_levels = [0.0001, 0.0001, 0.0001]
Mlr_levels = [0.0001, 0.0001, 0.0001]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float32)

# -----------------------------
# Utility functions
# -----------------------------

def load_video_frames(video_path, frame_start, frame_end):
    """
    Load frames from video_path using OpenCV.
    frame_start, frame_end are 1-based inclusive indices.
    Returns list of grayscale float32 images (H x W) normalized to [0,1].
    """
    assert os.path.exists(video_path), f"Video not found: {video_path}"
    cap = cv2.VideoCapture(video_path)
    frames = []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for idx in range(frame_start - 1, frame_end):
        if idx < 0 or idx >= total:
            raise IndexError(f"Requested frame {idx+1} out of bounds (total {total})")
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            raise RuntimeError(f"Failed to read frame {idx+1} from {video_path}")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        frames.append(gray)
    cap.release()
    return frames


def crop_roi(img, xmin, xmax, ymin, ymax):
    return img[ymin:ymax, xmin:xmax]


def build_image_pyramid(img, scales):
    """
    img: numpy array HxW (grayscale)
    scales: list of scale factors (e.g., [0.25, 0.5, 1.0])
    returns list of torch tensors on device (1,1,H,W) floats
    """
    pyram = []
    for s in scales:
        if s == 1.0:
            img_s = img
        else:
            h_new = max(1, int(round(img.shape[0] * s)))
            w_new = max(1, int(round(img.shape[1] * s)))
            img_s = cv2.resize(img, (w_new, h_new), interpolation=cv2.INTER_LINEAR)
        t = torch.from_numpy(img_s).unsqueeze(0).unsqueeze(0).to(device)  # (1,1,H,W)
        pyram.append(t)
    return pyram


def make_radial_masks(roi_h, roi_w, center, radii):
    """
    Build masks for rectangular annular regions.
    For each radius r, define a mask for pixels with distance from center in (r-2.5, r+2.5].
    Returns list of boolean masks shaped (H,W).
    """
    yy = np.arange(roi_h) - center[1]
    xx = np.arange(roi_w) - center[0]
    X, Y = np.meshgrid(xx, yy)
    R = np.sqrt(X**2 + Y**2)
    masks = []
    step = radii[0]  # 5
    for i, r in enumerate(radii):
        r_min = r - step / 2
        r_max = r + step / 2
        if r_min < 0:
            r_min = 0.0
        mask = (R > r_min) & (R <= r_max)
        masks.append(mask)
    return masks


def rotation_grid_from_theta(theta_map, center, H, W):
    """
    Build a grid tensor for grid_sample using rotation per-pixel.
    theta_map: torch tensor (H,W) giving rotation angles in radians
    center: (cx, cy) in pixel coordinates
    returns grid of shape (1,H,W,2) normalized to [-1,1]
    """
    ys = torch.arange(0, H, device=theta_map.device, dtype=theta_map.dtype)
    xs = torch.arange(0, W, device=theta_map.device, dtype=theta_map.dtype)
    Y, X = torch.meshgrid(ys, xs, indexing='ij')
    Xr = X - center[0]
    Yr = Y - center[1]
    ct = torch.cos(theta_map)
    st = torch.sin(theta_map)
    Xw = ct * Xr - st * Yr + center[0]
    Yw = st * Xr + ct * Yr + center[1]
    Hf = float(H)
    Wf = float(W)
    xn = 2.0 * (Xw / (Wf - 1) - 0.5)
    yn = 2.0 * (Yw / (Hf - 1) - 0.5)
    grid = torch.stack((xn, yn), dim=-1).unsqueeze(0)  # (1,H,W,2)
    return grid


def mse_np(a, b):
    return np.mean((a - b) ** 2)


# -----------------------------
# Bayesian Prior Fusion
# -----------------------------

def fuse_gaussian_priors(mu1, logvar1, mu2, logvar2):
    """
    Fuse two Gaussian distributions via product-of-Gaussians (Bayesian multiplication).

    Given:
        q1(theta) = N(mu1, sigma1^2)   <- posterior from previous frame
        q2(theta) = N(mu2, sigma2^2)   <- posterior from coarser pyramid level

    The fused prior is:
        p(theta) ∝ q1(theta) * q2(theta) = N(mu_fused, sigma_fused^2)

    where precision (1/variance) simply adds:
        prec_fused  = prec1 + prec2
        sigma_fused = 1 / prec_fused
        mu_fused    = sigma_fused^2 * (mu1 * prec1 + mu2 * prec2)

    This is the correct Bayesian way to integrate two independent sources of
    information about the same latent variable theta.

    Args:
        mu1      : (num_radii,) mean from source 1
        logvar1  : (num_radii,) log-variance from source 1
        mu2      : (num_radii,) mean from source 2
        logvar2  : (num_radii,) log-variance from source 2

    Returns:
        mu_fused     : (num_radii,) fused mean
        logvar_fused : (num_radii,) fused log-variance
    """
    var1 = torch.exp(logvar1)
    var2 = torch.exp(logvar2)

    prec1 = 1.0 / (var1 + 1e-8)
    prec2 = 1.0 / (var2 + 1e-8)

    prec_fused = prec1 + prec2

    var_fused    = 1.0 / (prec_fused + 1e-8)
    mu_fused     = (mu1 * prec1 + mu2 * prec2) / (prec_fused + 1e-8)
    logvar_fused = torch.log(var_fused + 1e-8)

    return mu_fused, logvar_fused


# -----------------------------
# Core Model
# -----------------------------

class IndependentRotEstimator:
    """
    Independent rotation estimator where each radius has its own rotation parameter.
    No basis functions - each radius is independent.
    """
    def __init__(self, roi_h, roi_w, radii, device):
        self.roi_h     = roi_h
        self.roi_w     = roi_w
        self.radii     = radii
        self.num_radii = len(radii)
        self.device    = device

    def theta_from_params(self, theta_params):
        """
        theta_params: (num_radii,) torch tensor
        Returns theta_params directly (identity mapping).
        """
        return theta_params

    def theta_map_from_theta_r(self, theta_r, radial_masks):
        """
        Create a full theta map by assigning each radius band its rotation value.
        """
        H, W = radial_masks[0].shape
        theta_map = torch.zeros((H, W), dtype=torch.float32, device=self.device)
        for i, mask in enumerate(radial_masks):
            if np.any(mask):
                mask_t = torch.from_numpy(mask).to(self.device)
                theta_map[mask_t] = theta_r[i]
        return theta_map


# -----------------------------
# Main processing
# -----------------------------

def main():
    print("\n" + "=" * 60)
    print("RUNNING INDEPENDENT ROTATION MODEL")
    print("Each radius has its own independent rotation parameter")
    print("Bayesian prior fusion: product-of-Gaussians from")
    print("  (1) previous frame posterior")
    print("  (2) coarser pyramid level posterior")
    print("=" * 60 + "\n")

    frames     = load_video_frames(video_path, FRAME_START, FRAME_END)
    num_frames = len(frames)
    frame_pairs = [(i, i + 1) for i in range(num_frames - 1)]
    num_pairs   = len(frame_pairs)

    frame0 = frames[0]
    roi0   = crop_roi(frame0, xmin, xmax, ymin, ymax)
    roi_h, roi_w = roi0.shape
    roi_center_x = int(round((xmax - xmin) / 2.0))
    roi_center_y = int(round((ymax - ymin) / 2.0))
    roi_center   = (roi_center_x, roi_center_y)

    radial_masks = make_radial_masks(roi_h, roi_w, roi_center, radii)
    estimator    = IndependentRotEstimator(roi_h, roi_w, radii, device)

    if not os.path.exists(csv_init_angles):
        raise FileNotFoundError(f"Initial angles CSV not found at {csv_init_angles}")

    init_df = pd.read_csv(csv_init_angles)

    if init_df.shape[1] >= num_radii + 1:
        init_angles_arr = init_df.iloc[0, 1:num_radii + 1].astype(float).values.astype(np.float32)
    else:
        possible_vals = pd.to_numeric(init_df.iloc[:, 1], errors='coerce')
        if possible_vals.notna().sum() >= num_radii:
            init_angles_arr = possible_vals.dropna().values[:num_radii].astype(np.float32)
        else:
            raise ValueError(
                "Initial angles CSV must contain at least 20 numeric rectangle values "
                "in columns 1..20 (after skipping 'Frame' column)."
            )

    mu_theta_init = torch.from_numpy(init_angles_arr).to(device)

    all_mu_theta_r_per_pair      = np.zeros((num_pairs, num_radii), dtype=np.float32)
    all_logvar_theta_r_per_pair  = np.zeros((num_pairs, num_radii), dtype=np.float32)
    centers_traj                 = np.zeros((num_pairs, 2),         dtype=np.float32)
    noise_variances_by_level     = [np.zeros(OUTER_EM_ITERS, dtype=np.float32) for _ in range(num_levels)]
    mse_by_level                 = [np.zeros(OUTER_EM_ITERS, dtype=np.float32) for _ in range(num_levels)]
    elbo_by_level                = [np.zeros(OUTER_EM_ITERS, dtype=np.float32) for _ in range(num_levels)]

    sigma_n_fixed_levels = [None] * num_levels

    for pidx, (i_f, j_f) in enumerate(frame_pairs):
        print(f"\nProcessing pair {pidx + 1}/{num_pairs}: "
              f"frame {FRAME_START + i_f} -> {FRAME_START + j_f}")

        f1_full = crop_roi(frames[i_f], xmin, xmax, ymin, ymax)
        f2_full = crop_roi(frames[j_f], xmin, xmax, ymin, ymax)

        pyr_f1 = build_image_pyramid(f1_full, pyramid_scales)
        pyr_f2 = build_image_pyramid(f2_full, pyramid_scales)

        per_level_params = []

        for level_idx, scale in enumerate(pyramid_scales):
            mu_theta     = mu_theta_init.clone().float()
            logvar_theta = torch.log(torch.ones_like(mu_theta) * (1.0 ** 2))

            sigma_n      = torch.tensor(0.02, dtype=torch.float32, device=device, requires_grad=True)
            c_x          = roi_center[0] * scale
            c_y          = roi_center[1] * scale
            center_param = torch.tensor([c_x, c_y], dtype=torch.float32, device=device, requires_grad=True)

            per_level_params.append({
                "mu_theta":     mu_theta.detach().clone().requires_grad_(True),
                "logvar_theta": logvar_theta.detach().clone().requires_grad_(True),
                "sigma_n":      sigma_n,
                "center":       center_param
            })

        if pidx > 0:
            prev_mu_theta     = torch.from_numpy(all_mu_theta_r_per_pair[pidx - 1, :]).to(device)
            prev_logvar_theta = torch.from_numpy(all_logvar_theta_r_per_pair[pidx - 1, :]).to(device)

            for level_idx in range(num_levels):
                per_level_params[level_idx]["mu_theta"]     = prev_mu_theta.clone().detach().requires_grad_(True)
                per_level_params[level_idx]["logvar_theta"] = prev_logvar_theta.clone().detach().requires_grad_(True)
                c_x = roi_center[0] * pyramid_scales[level_idx]
                c_y = roi_center[1] * pyramid_scales[level_idx]
                per_level_params[level_idx]["center"] = torch.tensor(
                    [c_x, c_y], dtype=torch.float32, device=device, requires_grad=True
                )
                per_level_params[level_idx]["sigma_n"] = torch.tensor(
                    0.02, dtype=torch.float32, device=device, requires_grad=True
                )

        for level_idx, scale in enumerate(pyramid_scales):
            print(f"  Level {level_idx} scale {scale}")

            f1_t = pyr_f1[level_idx].to(device)
            f2_t = pyr_f2[level_idx].to(device)
            Hlvl = f1_t.shape[2]
            Wlvl = f1_t.shape[3]

            if scale != 1.0:
                scaled_masks = []
                for mask in radial_masks:
                    mask_uint8 = (mask.astype(np.uint8) * 255)
                    mask_rs    = cv2.resize(mask_uint8, (Wlvl, Hlvl), interpolation=cv2.INTER_NEAREST)
                    scaled_masks.append(mask_rs.astype(bool))
            else:
                scaled_masks = radial_masks

            params = per_level_params[level_idx]

            freeze_mu_for_first_pair = (pidx == 0)

            mu_theta     = params["mu_theta"].requires_grad_(True)
            logvar_theta = params["logvar_theta"].requires_grad_(True)
            sigma_n      = params["sigma_n"].requires_grad_(True)
            center_param = params["center"].requires_grad_(True)

            params["mu_theta"]     = mu_theta
            params["logvar_theta"] = logvar_theta
            params["sigma_n"]      = sigma_n
            params["center"]       = center_param

            optimizerE = torch.optim.Adam([mu_theta, logvar_theta], lr=Elr_levels[level_idx])
            optimizerM = torch.optim.Adam([center_param, sigma_n],  lr=Mlr_levels[level_idx])

            if (sigma_n_fixed_levels[level_idx] is not None) and (pidx > 0):
                fixed_sigma_val    = float(sigma_n_fixed_levels[level_idx])
                sigma_n            = torch.tensor(fixed_sigma_val, dtype=torch.float32,
                                                  device=device, requires_grad=False)
                params["sigma_n"]  = sigma_n
                optimizerM         = torch.optim.Adam([center_param], lr=Mlr_levels[level_idx])

            for outer in range(OUTER_EM_ITERS):

                for e_it in range(E_UPDATES):
                    optimizerE.zero_grad()

                    std_theta     = torch.exp(0.5 * logvar_theta)
                    eps           = torch.randn((S_mc, mu_theta.shape[0]),
                                                device=device, dtype=mu_theta.dtype)
                    theta_samples = mu_theta.unsqueeze(0) + std_theta.unsqueeze(0) * eps

                    if freeze_mu_for_first_pair:
                        mu_theta_det  = mu_theta.detach()
                        theta_samples = mu_theta_det.unsqueeze(0) + std_theta.unsqueeze(0) * eps

                    losses = []
                    for s in range(S_mc):
                        theta_r   = theta_samples[s]
                        theta_map = estimator.theta_map_from_theta_r(theta_r, scaled_masks)
                        grid      = rotation_grid_from_theta(
                            theta_map, (center_param[0], center_param[1]), Hlvl, Wlvl
                        )
                        f2_warp   = F.grid_sample(f2_t, grid, mode='bilinear',
                                                   padding_mode='border', align_corners=True)
                        res       = f1_t - f2_warp
                        losses.append((res ** 2).sum())

                    mc_se = torch.stack(losses).mean()

                    sigma_n_param = params["sigma_n"]
                    loglik        = -0.5 * mc_se / (sigma_n_param ** 2)
                    kl            = torch.tensor(0.0, device=device)
                    elbo          = loglik - kl
                    loss          = -elbo
                    loss.backward()

                    if freeze_mu_for_first_pair:
                        if mu_theta.grad is not None:
                            mu_theta.grad.zero_()

                    optimizerE.step()

                for m_it in range(M_UPDATES):
                    optimizerM.zero_grad()

                    std_theta     = torch.exp(0.5 * logvar_theta)
                    eps           = torch.randn((S_mc, mu_theta.shape[0]), device=device)
                    theta_samples = mu_theta.unsqueeze(0) + std_theta.unsqueeze(0) * eps

                    se_list = []
                    for s in range(S_mc):
                        theta_r   = theta_samples[s]
                        theta_map = estimator.theta_map_from_theta_r(theta_r, scaled_masks)
                        grid      = rotation_grid_from_theta(
                            theta_map, (center_param[0], center_param[1]), Hlvl, Wlvl
                        )
                        f2_warp   = F.grid_sample(f2_t, grid, mode='bilinear',
                                                   padding_mode='border', align_corners=True)
                        res       = f1_t - f2_warp
                        se_list.append((res ** 2).sum())

                    mc_se        = torch.stack(se_list).mean()
                    loss_center  = mc_se
                    loss_center.backward()
                    optimizerM.step()

                with torch.no_grad():
                    num_pixels        = Hlvl * Wlvl
                    sigma_n2          = mc_se / num_pixels
                    params["sigma_n"] = torch.sqrt(sigma_n2)

                with torch.no_grad():
                    std_theta     = torch.exp(0.5 * logvar_theta)
                    eps           = torch.randn((S_mc, mu_theta.shape[0]),
                                                device=device, dtype=mu_theta.dtype)
                    theta_samples = mu_theta.unsqueeze(0) + std_theta.unsqueeze(0) * eps

                    losses_log = []
                    for s in range(S_mc):
                        theta_r   = theta_samples[s]
                        theta_map = estimator.theta_map_from_theta_r(theta_r, scaled_masks)
                        grid      = rotation_grid_from_theta(
                            theta_map, (center_param[0], center_param[1]), Hlvl, Wlvl
                        )
                        f2_warp   = F.grid_sample(f2_t, grid, mode='bilinear',
                                                   padding_mode='border', align_corners=True)
                        res       = f1_t - f2_warp
                        losses_log.append((res ** 2).sum())

                    mc_se_log  = float(torch.stack(losses_log).mean().cpu().numpy())
                    num_pixels = float(Hlvl * Wlvl)
                    mse_val    = mc_se_log / num_pixels

                    if isinstance(params["sigma_n"], torch.Tensor) and not params["sigma_n"].requires_grad:
                        sigma_n_val = params["sigma_n"].item()
                    else:
                        sigma_n_val = float(sigma_n.detach().cpu().numpy())

                    loglik_val = -0.5 * mc_se_log / (sigma_n_val ** 2)
                    elbo_val   = loglik_val

                    noise_variances_by_level[level_idx][outer] = sigma_n_val
                    mse_by_level[level_idx][outer]             = mse_val
                    elbo_by_level[level_idx][outer]            = elbo_val

                if (outer + 1) % 5 == 0:
                    print(f"    outer {outer+1}/{OUTER_EM_ITERS}: "
                          f"sigma_n={sigma_n_val:.6f}, mse={mse_val:.6e}, elbo={elbo_val:.4f}")

            per_level_params[level_idx]["mu_theta"]     = mu_theta.detach()
            per_level_params[level_idx]["logvar_theta"] = logvar_theta.detach()
            per_level_params[level_idx]["sigma_n"]      = (sigma_n.detach()
                                                            if isinstance(sigma_n, torch.Tensor)
                                                            else sigma_n)
            per_level_params[level_idx]["center"]       = center_param.detach()

            if pidx == 0:
                sigma_n_fixed_levels[level_idx] = float(
                    per_level_params[level_idx]["sigma_n"].cpu().numpy()
                )

            if level_idx < num_levels - 1:
                mu_coarse     = per_level_params[level_idx]["mu_theta"]
                logvar_coarse = per_level_params[level_idx]["logvar_theta"]

                mu_prev_frame     = per_level_params[level_idx + 1]["mu_theta"].detach()
                logvar_prev_frame = per_level_params[level_idx + 1]["logvar_theta"].detach()

                mu_fused, logvar_fused = fuse_gaussian_priors(
                    mu_coarse,     logvar_coarse,
                    mu_prev_frame, logvar_prev_frame
                )

                per_level_params[level_idx + 1]["mu_theta"]     = mu_fused.detach().requires_grad_(True)
                per_level_params[level_idx + 1]["logvar_theta"] = logvar_fused.detach().requires_grad_(True)

                scale_ratio = pyramid_scales[level_idx + 1] / pyramid_scales[level_idx]
                cx = per_level_params[level_idx]["center"][0].cpu().item() * scale_ratio
                cy = per_level_params[level_idx]["center"][1].cpu().item() * scale_ratio
                per_level_params[level_idx + 1]["center"] = torch.tensor(
                    [cx, cy], dtype=torch.float32, device=device, requires_grad=True
                )
                per_level_params[level_idx + 1]["sigma_n"] = per_level_params[level_idx]["sigma_n"]

        mu_theta_fin     = per_level_params[-1]["mu_theta"].cpu().numpy()
        logvar_theta_fin = per_level_params[-1]["logvar_theta"].cpu().numpy()

        all_mu_theta_r_per_pair[pidx, :]     = mu_theta_fin.copy()
        all_logvar_theta_r_per_pair[pidx, :] = logvar_theta_fin.copy()
        centers_traj[pidx, :]                = per_level_params[-1]["center"].cpu().numpy()

        print(f"  Finished pair {pidx + 1}: "
              f"center={centers_traj[pidx, :]}, "
              f"first 5 theta values={mu_theta_fin[:5]}")

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    frame_pair_indices = np.arange(FRAME_START, FRAME_START + num_pairs)

    fig1, ax1 = plt.subplots(figsize=(14, 8))
    for ridx in range(0, num_radii, 1):
        r     = radii[ridx]
        yvals = all_mu_theta_r_per_pair[:, ridx]
        ax1.plot(frame_pair_indices, yvals,
                 marker='o', linewidth=2, markersize=6, label=f"r={int(r)}")
    ax1.set_xlabel("Frame index")
    ax1.set_ylabel("Rotation angle (radians)")
    ax1.set_title("Independent Rotation per Radius (Bayesian Prior Fusion)")
    ax1.axhline(0, linestyle='--', linewidth=1)
    ax1.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
    ax1.tick_params(axis='x', labelsize=18)
    ax1.tick_params(axis='y', labelsize=18)
    plt.tight_layout()
    plt.show()

    metrics = [
        ("Noise Variance",        noise_variances_by_level),
        ("MSE",                   mse_by_level),
        ("ELBO (Log-Likelihood)", elbo_by_level)
    ]

    for metric_name, metric_data in metrics:
        fig = plt.figure(figsize=(14, 7))
        ax_main = plt.subplot2grid((3, 2), (0, 0), rowspan=3)
        colors  = ['C0', 'C1', 'C2']
        for lvl in range(num_levels):
            ax_main.plot(np.arange(1, OUTER_EM_ITERS + 1), metric_data[lvl],
                         color=colors[lvl], label=f"Level {lvl}")
        ax_main.set_xlabel("Outer EM iteration")
        ax_main.set_ylabel(metric_name)
        ax_main.set_title(f"{metric_name} across pyramid levels")
        ax_main.legend()

        for lvl in range(num_levels):
            ax = plt.subplot2grid((3, 2), (lvl, 1))
            ax.plot(np.arange(1, OUTER_EM_ITERS + 1), metric_data[lvl], color=colors[lvl])
            ax.set_title(f"Level {lvl}")
            ax.set_xlabel("Iteration")

        plt.tight_layout()
        plt.show()

    centers   = centers_traj.copy()
    frame_ids = np.arange(FRAME_START, FRAME_START + num_pairs)

    fig_traj, ax_traj = plt.subplots(figsize=(7, 8))
    sc = ax_traj.scatter(centers[:, 0], centers[:, 1],
                         c=frame_ids, cmap='viridis', s=60, zorder=3)
    ax_traj.plot(centers[:, 0], centers[:, 1],
                 color='black', linewidth=1.5, label='Center trajectory', zorder=2)
    ax_traj.add_patch(
        plt.Rectangle((0, 0), roi_w, roi_h,
                       fill=False, edgecolor='red', linestyle='--',
                       linewidth=2, label='ROI')
    )
    ax_traj.set_title("Rotation Center Trajectory - Independent Rotation")
    ax_traj.set_xlabel("x (pixels)")
    ax_traj.set_ylabel("y (pixels)")
    ax_traj.invert_yaxis()
    ax_traj.set_aspect('equal', adjustable='box')
    cbar = plt.colorbar(sc, ax=ax_traj)
    cbar.set_label("Frame pair")
    ax_traj.legend()
    plt.tight_layout()
    plt.show()

    fig_var, ax_var = plt.subplots(figsize=(14, 6))
    for ridx in range(0, num_radii, 2):
        r        = radii[ridx]
        var_vals = np.exp(all_logvar_theta_r_per_pair[:, ridx])
        ax_var.plot(frame_pair_indices, var_vals,
                    marker='s', linewidth=1.5, markersize=4, label=f"r={int(r)}")
    ax_var.set_xlabel("Frame index")
    ax_var.set_ylabel("Posterior variance (rad²)")
    ax_var.set_title("Posterior Variance per Radius over Time\n"
                     "(should decrease or stay low with Bayesian fusion)")
    ax_var.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
    ax_var.tick_params(axis='x', labelsize=18)
    ax_var.tick_params(axis='y', labelsize=18)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
