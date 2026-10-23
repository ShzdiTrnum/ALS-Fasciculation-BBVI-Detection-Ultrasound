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

# =====================================================================
# REAL-DATA experiment (no ground truth available)
# ---------------------------------------------------------------------
# This is the REAL patient-ultrasound counterpart of the synthetic script.
# The real data has NO ground-truth rotation angles, so we CANNOT compute
# angle error / center error against truth. Instead we evaluate:
#   * MSE (photometric alignment error) -- the paper's unsupervised metric
#   * smoothness of the recovered theta(r) curve (a property of the estimate)
#   * frame-to-frame stability of theta(r)
# The optical-flow CSV is used ONLY as an initialisation seed (NOT as truth).
#
#   STRUCTURED = False -> independent / unstructured (each radius free)
#   STRUCTURED = True  -> structured-in-r (per-radius mean/logvar from a
#                         low-order cosine basis in r)
#   FRAME_STEP = 1     -> consecutive frames
#   FRAME_STEP = 3     -> skipped / distant frames (large-motion stress)
#
# Run all four combinations: {False, True} x {1, 3}
# =====================================================================
STRUCTURED = True        # <--- toggle this
U_BASIS    = 6            # cosine basis terms (structured arm). U << num_radii (=20)

# -----------------------------
# User-specified paths & params
# -----------------------------
video_path      = "data/ALS - Trim20s.mp4"
# Optical-flow-derived initial angles -- SEED ONLY, not ground truth.
csv_init_angles = "data/Original_Groundtruth_Values.csv"

# Per-run summary (unsupervised metrics) appended here, one row per run.
RESULTS_CSV     = "outputs/real_results.csv"
# Per-run full arrays for the separate comparison script.
PERPAIR_DIR     = "outputs/real_perpair_runs"

FRAME_START = 6    # inclusive 1-based start frame
FRAME_END   = 24   # inclusive end frame
FRAME_STEP  = 3    # 1 = consecutive, 3 = skipped/distant

# ROI
xmin, xmax = 250, 500
ymin, ymax = 200, 450

# Radii: 5-pixel intervals up to 100 -> 20 rings
radii = np.arange(5, 101, 5)
num_radii = len(radii)

# Pyramid scales (coarse -> fine)
pyramid_scales = [0.25, 0.5, 1.0]
num_levels = len(pyramid_scales)

# BBVI / EM hyperparams
S_mc = 20
OUTER_EM_ITERS = 50
E_UPDATES = 30
M_UPDATES = 50

# Learning rates per level
Elr_levels = [0.0001, 0.0001, 0.0001]
Mlr_levels = [0.0001, 0.0001, 0.0001]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float32)

# --- added for the public release: command-line overrides + figure saving ---
from runconfig import configure
configure(globals())


# -----------------------------
# Utility functions
# -----------------------------

def load_video_frames(video_path, frame_start, frame_end):
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
    pyram = []
    for s in scales:
        if s == 1.0:
            img_s = img
        else:
            h_new = max(1, int(round(img.shape[0] * s)))
            w_new = max(1, int(round(img.shape[1] * s)))
            img_s = cv2.resize(img, (w_new, h_new), interpolation=cv2.INTER_LINEAR)
        t = torch.from_numpy(img_s).unsqueeze(0).unsqueeze(0).to(device)
        pyram.append(t)
    return pyram


def make_radial_masks(roi_h, roi_w, center, radii):
    yy = np.arange(roi_h) - center[1]
    xx = np.arange(roi_w) - center[0]
    X, Y = np.meshgrid(xx, yy)
    R = np.sqrt(X**2 + Y**2)
    masks = []
    step = radii[0]
    for i, r in enumerate(radii):
        r_min = r - step / 2
        r_max = r + step / 2
        if r_min < 0:
            r_min = 0.0
        mask = (R > r_min) & (R <= r_max)
        masks.append(mask)
    return masks


def rotation_grid_from_theta(theta_map, center, H, W):
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
    grid = torch.stack((xn, yn), dim=-1).unsqueeze(0)
    return grid


def mse_np(a, b):
    return np.mean((a - b) ** 2)


# -----------------------------
# Structured basis in r  (comment 1a, option A)
# -----------------------------

def build_cosine_basis(radii, U):
    """
    Orthonormal DCT-II cosine basis Phi (num_radii, U) evaluated at ring radii.
    Column 0 = constant; each further column adds one half-cosine 'wiggle' in r.
    mu(r) = Phi @ alpha is therefore smooth in r by construction.
    """
    R = len(radii)
    n = np.arange(R)
    Phi = np.zeros((R, U), dtype=np.float32)
    Phi[:, 0] = 1.0 / np.sqrt(R)
    for u in range(1, U):
        Phi[:, u] = np.sqrt(2.0 / R) * np.cos(np.pi * u * (n + 0.5) / R)
    return torch.from_numpy(Phi).to(device)


def fit_coeffs_to_angles(Phi, angles):
    """Least-squares fit: solve Phi @ c ~= angles. Returns (U,) tensor."""
    Phi_np = Phi.detach().cpu().numpy()
    ang_np = angles.detach().cpu().numpy()
    c, *_  = np.linalg.lstsq(Phi_np, ang_np, rcond=None)
    return torch.from_numpy(c.astype(np.float32)).to(device)


# -----------------------------
# Bayesian Prior Fusion (product-of-Gaussians)
# -----------------------------

def fuse_gaussian_priors(mu1, logvar1, mu2, logvar2):
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
# Unsupervised smoothness metric (no ground truth needed)
# -----------------------------

def radial_roughness(theta_curve):
    """
    Roughness of a theta(r) curve = mean squared 2nd difference along r.
    Lower = smoother. This is a property of the ESTIMATE itself; it needs
    no ground truth, so it is valid on real data. theta_curve: (num_radii,).
    """
    d2 = theta_curve[2:] - 2.0 * theta_curve[1:-1] + theta_curve[:-2]
    return float(np.mean(d2 ** 2))


# -----------------------------
# Core Model
# -----------------------------

class IndependentRotEstimator:
    """theta_map builder shared by both arms (parameterisation differs upstream)."""
    def __init__(self, roi_h, roi_w, radii, device):
        self.roi_h     = roi_h
        self.roi_w     = roi_w
        self.radii     = radii
        self.num_radii = len(radii)
        self.device    = device

    def theta_map_from_theta_r(self, theta_r, radial_masks):
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
    arm_name = "STRUCTURED (cosine basis in r)" if STRUCTURED else "UNSTRUCTURED (independent per radius)"
    print("\n" + "=" * 66)
    print(f"RUNNING {arm_name}  [REAL DATA -- no ground truth]")
    if STRUCTURED:
        print(f"  Basis: DCT-II cosine in r,  U = {U_BASIS}  ({U_BASIS} coeffs << {num_radii} rings)")
    print("Bayesian prior fusion: product-of-Gaussians across levels & frames")
    print(f"Frame sampling: FRAME_STEP={FRAME_STEP} (pairs: i -> i+{FRAME_STEP})")
    print("Evaluation: MSE + radial smoothness + frame-to-frame stability")
    print("=" * 66 + "\n")

    # ------------------------------------------------------------------
    # Load frames
    # ------------------------------------------------------------------
    frames     = load_video_frames(video_path, FRAME_START, FRAME_END)
    num_frames = len(frames)

    frame_pairs = [(i, i + FRAME_STEP) for i in range(num_frames - FRAME_STEP)]
    num_pairs   = len(frame_pairs)

    print(f"Total frames loaded : {num_frames}")
    print(f"Total frame pairs   : {num_pairs} (step={FRAME_STEP})\n")

    # ------------------------------------------------------------------
    # ROI dimensions and center
    # ------------------------------------------------------------------
    frame0 = frames[0]
    roi0   = crop_roi(frame0, xmin, xmax, ymin, ymax)
    roi_h, roi_w = roi0.shape
    roi_center_x = int(round((xmax - xmin) / 2.0))
    roi_center_y = int(round((ymax - ymin) / 2.0))
    roi_center   = (roi_center_x, roi_center_y)

    radial_masks = make_radial_masks(roi_h, roi_w, roi_center, radii)
    estimator = IndependentRotEstimator(roi_h, roi_w, radii, device)

    Phi = build_cosine_basis(radii, U_BASIS) if STRUCTURED else None

    # ------------------------------------------------------------------
    # Load initial rotation angles CSV (SEED ONLY -- from optical flow)
    # ------------------------------------------------------------------
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
            raise ValueError("Initial angles CSV must contain at least num_radii numeric values.")

    mu_theta_init = torch.from_numpy(init_angles_arr).to(device)

    if STRUCTURED:
        alpha_init = fit_coeffs_to_angles(Phi, mu_theta_init)
        logvar_seed_vec = torch.zeros_like(mu_theta_init)  # log(1)=0
        beta_init  = fit_coeffs_to_angles(Phi, logvar_seed_vec)

    # ------------------------------------------------------------------
    # Storage
    # ------------------------------------------------------------------
    all_mu_theta_r_per_pair     = np.zeros((num_pairs, num_radii), dtype=np.float32)
    all_logvar_theta_r_per_pair = np.zeros((num_pairs, num_radii), dtype=np.float32)
    centers_traj                = np.zeros((num_pairs, 2),         dtype=np.float32)
    final_mse_per_pair          = np.zeros(num_pairs,              dtype=np.float32)
    noise_variances_by_level    = [np.zeros(OUTER_EM_ITERS, dtype=np.float32) for _ in range(num_levels)]
    mse_by_level                = [np.zeros(OUTER_EM_ITERS, dtype=np.float32) for _ in range(num_levels)]
    elbo_by_level               = [np.zeros(OUTER_EM_ITERS, dtype=np.float32) for _ in range(num_levels)]

    sigma_n_fixed_levels = [None] * num_levels

    # ------------------------------------------------------------------
    # Iterate over frame pairs
    # ------------------------------------------------------------------
    for pidx, (i_f, j_f) in enumerate(frame_pairs):
        print(f"\nProcessing pair {pidx + 1}/{num_pairs}: "
              f"frame {FRAME_START + i_f} -> {FRAME_START + j_f} (gap={j_f - i_f})")

        f1_full = crop_roi(frames[i_f], xmin, xmax, ymin, ymax)
        f2_full = crop_roi(frames[j_f], xmin, xmax, ymin, ymax)

        pyr_f1 = build_image_pyramid(f1_full, pyramid_scales)
        pyr_f2 = build_image_pyramid(f2_full, pyramid_scales)

        # ---- per-level init ----
        per_level_params = []
        for level_idx, scale in enumerate(pyramid_scales):
            sigma_n      = torch.tensor(0.02, dtype=torch.float32, device=device, requires_grad=True)
            c_x          = roi_center[0] * scale
            c_y          = roi_center[1] * scale
            center_param = torch.tensor([c_x, c_y], dtype=torch.float32, device=device, requires_grad=True)
            entry = {"sigma_n": sigma_n, "center": center_param}
            if STRUCTURED:
                entry["alpha"] = alpha_init.detach().clone().requires_grad_(True)
                entry["beta"]  = beta_init.detach().clone().requires_grad_(True)
            else:
                mu_theta     = mu_theta_init.clone().float()
                logvar_theta = torch.log(torch.ones_like(mu_theta) * (1.0 ** 2))
                entry["mu_theta"]     = mu_theta.detach().clone().requires_grad_(True)
                entry["logvar_theta"] = logvar_theta.detach().clone().requires_grad_(True)
            per_level_params.append(entry)

        # ---- seed from previous frame posterior ----
        if pidx > 0:
            prev_mu_theta     = torch.from_numpy(all_mu_theta_r_per_pair[pidx - 1, :]).to(device)
            prev_logvar_theta = torch.from_numpy(all_logvar_theta_r_per_pair[pidx - 1, :]).to(device)
            if STRUCTURED:
                prev_alpha = fit_coeffs_to_angles(Phi, prev_mu_theta)
                prev_beta  = fit_coeffs_to_angles(Phi, prev_logvar_theta)
            for level_idx in range(num_levels):
                if STRUCTURED:
                    per_level_params[level_idx]["alpha"] = prev_alpha.clone().detach().requires_grad_(True)
                    per_level_params[level_idx]["beta"]  = prev_beta.clone().detach().requires_grad_(True)
                else:
                    per_level_params[level_idx]["mu_theta"]     = prev_mu_theta.clone().detach().requires_grad_(True)
                    per_level_params[level_idx]["logvar_theta"] = prev_logvar_theta.clone().detach().requires_grad_(True)
                c_x = roi_center[0] * pyramid_scales[level_idx]
                c_y = roi_center[1] * pyramid_scales[level_idx]
                per_level_params[level_idx]["center"] = torch.tensor(
                    [c_x, c_y], dtype=torch.float32, device=device, requires_grad=True)
                per_level_params[level_idx]["sigma_n"] = torch.tensor(
                    0.02, dtype=torch.float32, device=device, requires_grad=True)

        # ---- pyramid levels coarse -> fine ----
        for level_idx, scale in enumerate(pyramid_scales):
            print(f"  Level {level_idx} scale {scale}")

            f1_t = pyr_f1[level_idx].to(device)
            f2_t = pyr_f2[level_idx].to(device)
            Hlvl = f1_t.shape[2]; Wlvl = f1_t.shape[3]

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

            if STRUCTURED:
                alpha = params["alpha"].requires_grad_(True)
                beta  = params["beta"].requires_grad_(True)
                params["alpha"] = alpha; params["beta"] = beta
                e_param_list = [alpha, beta]
            else:
                mu_theta     = params["mu_theta"].requires_grad_(True)
                logvar_theta = params["logvar_theta"].requires_grad_(True)
                params["mu_theta"] = mu_theta; params["logvar_theta"] = logvar_theta
                e_param_list = [mu_theta, logvar_theta]

            sigma_n      = params["sigma_n"].requires_grad_(True)
            center_param = params["center"].requires_grad_(True)
            params["sigma_n"] = sigma_n; params["center"] = center_param

            optimizerE = torch.optim.Adam(e_param_list, lr=Elr_levels[level_idx])
            optimizerM = torch.optim.Adam([center_param, sigma_n], lr=Mlr_levels[level_idx])

            if (sigma_n_fixed_levels[level_idx] is not None) and (pidx > 0):
                fixed_sigma_val   = float(sigma_n_fixed_levels[level_idx])
                sigma_n           = torch.tensor(fixed_sigma_val, dtype=torch.float32,
                                                 device=device, requires_grad=False)
                params["sigma_n"] = sigma_n
                optimizerM        = torch.optim.Adam([center_param], lr=Mlr_levels[level_idx])

            def current_mu_logvar():
                if STRUCTURED:
                    return (Phi @ params["alpha"]), (Phi @ params["beta"])
                else:
                    return params["mu_theta"], params["logvar_theta"]

            for outer in range(OUTER_EM_ITERS):
                # ---- E-step ----
                for e_it in range(E_UPDATES):
                    optimizerE.zero_grad()
                    mu_theta_cur, logvar_theta_cur = current_mu_logvar()
                    std_theta     = torch.exp(0.5 * logvar_theta_cur)
                    eps           = torch.randn((S_mc, num_radii), device=device, dtype=torch.float32)
                    theta_samples = mu_theta_cur.unsqueeze(0) + std_theta.unsqueeze(0) * eps
                    if freeze_mu_for_first_pair:
                        mu_theta_det  = mu_theta_cur.detach()
                        theta_samples = mu_theta_det.unsqueeze(0) + std_theta.unsqueeze(0) * eps
                    losses = []
                    for s in range(S_mc):
                        theta_r   = theta_samples[s]
                        theta_map = estimator.theta_map_from_theta_r(theta_r, scaled_masks)
                        grid      = rotation_grid_from_theta(
                            theta_map, (center_param[0], center_param[1]), Hlvl, Wlvl)
                        f2_warp   = F.grid_sample(f2_t, grid, mode='bilinear',
                                                   padding_mode='border', align_corners=True)
                        res       = f1_t - f2_warp
                        losses.append((res ** 2).sum())
                    mc_se         = torch.stack(losses).mean()
                    sigma_n_param = params["sigma_n"]
                    loglik        = -0.5 * mc_se / (sigma_n_param ** 2)
                    kl            = torch.tensor(0.0, device=device)
                    elbo          = loglik - kl
                    loss          = -elbo
                    loss.backward()
                    if freeze_mu_for_first_pair:
                        if STRUCTURED:
                            if params["alpha"].grad is not None:
                                params["alpha"].grad.zero_()
                        else:
                            if params["mu_theta"].grad is not None:
                                params["mu_theta"].grad.zero_()
                    optimizerE.step()

                # ---- M-step ----
                for m_it in range(M_UPDATES):
                    optimizerM.zero_grad()
                    mu_theta_cur, logvar_theta_cur = current_mu_logvar()
                    std_theta     = torch.exp(0.5 * logvar_theta_cur)
                    eps           = torch.randn((S_mc, num_radii), device=device)
                    theta_samples = mu_theta_cur.unsqueeze(0) + std_theta.unsqueeze(0) * eps
                    se_list = []
                    for s in range(S_mc):
                        theta_r   = theta_samples[s]
                        theta_map = estimator.theta_map_from_theta_r(theta_r, scaled_masks)
                        grid      = rotation_grid_from_theta(
                            theta_map, (center_param[0], center_param[1]), Hlvl, Wlvl)
                        f2_warp   = F.grid_sample(f2_t, grid, mode='bilinear',
                                                   padding_mode='border', align_corners=True)
                        res       = f1_t - f2_warp
                        se_list.append((res ** 2).sum())
                    mc_se       = torch.stack(se_list).mean()
                    loss_center = mc_se
                    loss_center.backward()
                    optimizerM.step()

                with torch.no_grad():
                    num_pixels        = Hlvl * Wlvl
                    sigma_n2          = mc_se / num_pixels
                    params["sigma_n"] = torch.sqrt(sigma_n2)

                # ---- logging ----
                with torch.no_grad():
                    mu_theta_cur, logvar_theta_cur = current_mu_logvar()
                    std_theta     = torch.exp(0.5 * logvar_theta_cur)
                    eps           = torch.randn((S_mc, num_radii), device=device, dtype=torch.float32)
                    theta_samples = mu_theta_cur.unsqueeze(0) + std_theta.unsqueeze(0) * eps
                    losses_log = []
                    for s in range(S_mc):
                        theta_r   = theta_samples[s]
                        theta_map = estimator.theta_map_from_theta_r(theta_r, scaled_masks)
                        grid      = rotation_grid_from_theta(
                            theta_map, (center_param[0], center_param[1]), Hlvl, Wlvl)
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

            # materialise final mu/logvar
            with torch.no_grad():
                if STRUCTURED:
                    mu_final     = (Phi @ params["alpha"]).detach()
                    logvar_final = (Phi @ params["beta"]).detach()
                else:
                    mu_final     = params["mu_theta"].detach()
                    logvar_final = params["logvar_theta"].detach()
            per_level_params[level_idx]["mu_theta"]     = mu_final
            per_level_params[level_idx]["logvar_theta"] = logvar_final
            per_level_params[level_idx]["sigma_n"]      = (sigma_n.detach()
                                                            if isinstance(sigma_n, torch.Tensor) else sigma_n)
            per_level_params[level_idx]["center"]       = center_param.detach()

            if pidx == 0:
                sigma_n_fixed_levels[level_idx] = float(
                    per_level_params[level_idx]["sigma_n"].cpu().numpy())

            # ---- coarse->fine Bayesian fusion ----
            if level_idx < num_levels - 1:
                mu_coarse     = per_level_params[level_idx]["mu_theta"]
                logvar_coarse = per_level_params[level_idx]["logvar_theta"]
                mu_prev_frame     = per_level_params[level_idx + 1]["mu_theta"].detach() \
                                    if "mu_theta" in per_level_params[level_idx + 1] else mu_coarse
                logvar_prev_frame = per_level_params[level_idx + 1]["logvar_theta"].detach() \
                                    if "logvar_theta" in per_level_params[level_idx + 1] else logvar_coarse
                mu_fused, logvar_fused = fuse_gaussian_priors(
                    mu_coarse, logvar_coarse, mu_prev_frame, logvar_prev_frame)
                if STRUCTURED:
                    nxt_alpha = fit_coeffs_to_angles(Phi, mu_fused)
                    nxt_beta  = fit_coeffs_to_angles(Phi, logvar_fused)
                    per_level_params[level_idx + 1]["alpha"] = nxt_alpha.detach().requires_grad_(True)
                    per_level_params[level_idx + 1]["beta"]  = nxt_beta.detach().requires_grad_(True)
                else:
                    per_level_params[level_idx + 1]["mu_theta"]     = mu_fused.detach().requires_grad_(True)
                    per_level_params[level_idx + 1]["logvar_theta"] = logvar_fused.detach().requires_grad_(True)
                scale_ratio = pyramid_scales[level_idx + 1] / pyramid_scales[level_idx]
                cx = per_level_params[level_idx]["center"][0].cpu().item() * scale_ratio
                cy = per_level_params[level_idx]["center"][1].cpu().item() * scale_ratio
                per_level_params[level_idx + 1]["center"] = torch.tensor(
                    [cx, cy], dtype=torch.float32, device=device, requires_grad=True)
                per_level_params[level_idx + 1]["sigma_n"] = per_level_params[level_idx]["sigma_n"]

        # ---- record finest-level results ----
        mu_theta_fin     = per_level_params[-1]["mu_theta"].cpu().numpy()
        logvar_theta_fin = per_level_params[-1]["logvar_theta"].cpu().numpy()
        all_mu_theta_r_per_pair[pidx, :]     = mu_theta_fin.copy()
        all_logvar_theta_r_per_pair[pidx, :] = logvar_theta_fin.copy()
        centers_traj[pidx, :]                = per_level_params[-1]["center"].cpu().numpy()
        final_mse_per_pair[pidx]             = mse_by_level[-1][-1]  # finest-level final MSE
        print(f"  Finished pair {pidx + 1}: center={centers_traj[pidx, :]}, "
              f"first 5 theta={mu_theta_fin[:5]}")

    # ==================================================================
    # UNSUPERVISED EVALUATION  (no ground truth on real data)
    # ==================================================================
    est = all_mu_theta_r_per_pair.copy()

    # (1) MSE -- photometric alignment error (paper's unsupervised metric)
    mean_mse = float(final_mse_per_pair.mean())

    # (2) radial smoothness -- roughness of theta(r) per pair (lower=smoother)
    roughness_per_pair = np.array([radial_roughness(est[p]) for p in range(num_pairs)],
                                  dtype=np.float32)
    mean_roughness = float(roughness_per_pair.mean())

    # (3) frame-to-frame stability -- how much theta(r) jitters between pairs
    #     (mean over radii of the std across pairs). Lower = more stable.
    if num_pairs > 1:
        stability = float(np.mean(np.std(est, axis=0)))
    else:
        stability = float('nan')

    print("\n" + "-" * 66)
    print(f"UNSUPERVISED METRICS  |  arm={'structured' if STRUCTURED else 'unstructured'}  "
          f"U={U_BASIS if STRUCTURED else '-'}  FRAME_STEP={FRAME_STEP}")
    print("-" * 66)
    print(f"  Mean MSE (photometric)      : {mean_mse:.6e}")
    print(f"  Mean radial roughness       : {mean_roughness:.6e}  (lower = smoother theta(r))")
    print(f"  Frame-to-frame stability    : {stability:.6e}  (lower = steadier across frames)")
    print("-" * 66)

    # ---- append summary row ----
    row = {
        "arm":          "structured" if STRUCTURED else "unstructured",
        "U":            U_BASIS if STRUCTURED else 0,
        "frame_step":   FRAME_STEP,
        "num_pairs":    num_pairs,
        "mean_mse":     mean_mse,
        "mean_roughness": mean_roughness,
        "stability":    stability,
    }
    try:
        if os.path.exists(RESULTS_CSV):
            df_prev = pd.read_csv(RESULTS_CSV)
            df_out  = pd.concat([df_prev, pd.DataFrame([row])], ignore_index=True)
        else:
            df_out  = pd.DataFrame([row])
        df_out.to_csv(RESULTS_CSV, index=False)
        print(f"[results] appended run to {RESULTS_CSV}")
    except Exception as e:
        print(f"[results] could not write {RESULTS_CSV}: {e}")

    # ---- dump full per-pair arrays (for the comparison script) ----
    try:
        os.makedirs(PERPAIR_DIR, exist_ok=True)
        arm_tag = "structured" if STRUCTURED else "unstructured"
        fname   = os.path.join(PERPAIR_DIR, f"{arm_tag}_step{FRAME_STEP}.npz")
        np.savez(
            fname,
            arm=arm_tag,
            U=(U_BASIS if STRUCTURED else 0),
            frame_step=FRAME_STEP,
            radii=radii,
            source_frames=np.array([FRAME_START + i_f for (i_f, _) in frame_pairs]),
            est_theta=est,                          # (num_pairs, num_radii) rad
            logvar_theta=all_logvar_theta_r_per_pair,
            centers=centers_traj,                   # (num_pairs, 2) ROI coords
            final_mse_per_pair=final_mse_per_pair,  # (num_pairs,)
            roughness_per_pair=roughness_per_pair,  # (num_pairs,)
            mse_curve_final_level=mse_by_level[-1], # (OUTER_EM_ITERS,)
        )
        print(f"[perpair] saved full arrays to {fname}")
    except Exception as e:
        print(f"[perpair] could not write per-pair arrays: {e}")

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------
    frame_pair_indices = np.array([FRAME_START + i_f for (i_f, _) in frame_pairs])

    # 1. Rotation angle vs frame for each radius
    fig1, ax1 = plt.subplots(figsize=(14, 8))
    for ridx in range(0, num_radii, 1):
        r     = radii[ridx]
        yvals = all_mu_theta_r_per_pair[:, ridx]
        ax1.plot(frame_pair_indices, yvals, marker='o', linewidth=2, markersize=6,
                 label=f"r={int(r)}")
    ax1.set_xlabel("Source frame index")
    ax1.set_ylabel("Rotation angle (radians)")
    ax1.set_title(f"Rotation per radius "
                  f"({'structured' if STRUCTURED else 'unstructured'}, step={FRAME_STEP})")
    ax1.axhline(0, linestyle='--', linewidth=1)
    ax1.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
    ax1.tick_params(axis='x', labelsize=18)
    ax1.tick_params(axis='y', labelsize=18)
    plt.tight_layout()
    plt.show()

    # 1b. NEW: recovered theta(r) curve for the highest-motion pair + roughness per frame.
    # (No ground truth to overlay; we show the estimate's shape and its smoothness.)
    motion_proxy = np.abs(est).mean(axis=1)   # proxy: larger mean |theta| = more motion
    show_pair = int(np.argmax(motion_proxy))
    show_src  = FRAME_START + frame_pairs[show_pair][0]
    fig_c, (axA, axB) = plt.subplots(1, 2, figsize=(16, 6))
    axA.plot(radii, est[show_pair, :], marker='o')
    axA.set_xlabel("Radius r (px)")
    axA.set_ylabel("Rotation angle (rad)")
    axA.set_title(f"Recovered theta(r), largest-motion pair (src {show_src})\n"
                  f"({'structured' if STRUCTURED else 'unstructured'}, step={FRAME_STEP})")
    axA.axhline(0, linestyle='--', linewidth=1)
    axB.plot(frame_pair_indices, roughness_per_pair, marker='o', color='C3')
    axB.set_xlabel("Source frame index")
    axB.set_ylabel("Radial roughness (rad^2)")
    axB.set_title("Smoothness of theta(r) per frame (lower = smoother)")
    plt.tight_layout()
    plt.show()

    # 2. Metrics plots
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

    # 3. Center trajectory (+ zoom)  -- no true center overlay (real data)
    centers = centers_traj.copy()
    fig_traj, ax_traj = plt.subplots(figsize=(9, 9))
    sc = ax_traj.scatter(centers[:, 0], centers[:, 1],
                         c=frame_pair_indices, cmap='viridis', s=60, zorder=3)
    ax_traj.plot(centers[:, 0], centers[:, 1], color='black', linewidth=1.5,
                 label='Connecting Center Trajectory', zorder=2)
    for i, (xi, yi) in enumerate(zip(centers[:, 0], centers[:, 1])):
        ax_traj.annotate(str(frame_pair_indices[i]), xy=(xi, yi), fontsize=9,
                         ha='center', va='bottom', xytext=(0, 5), textcoords='offset points')
    ax_traj.add_patch(plt.Rectangle((0, 0), roi_w, roi_h, fill=False, edgecolor='red',
                                    linestyle='--', linewidth=2, label='ROI'))
    ax_traj.set_title("Adaptive COR Trajectory within Same ROI")
    ax_traj.set_xlabel("X Coordinate"); ax_traj.set_ylabel("Y Coordinate")
    ax_traj.invert_yaxis()
    ax_traj.set_aspect('equal', adjustable='box')
    cbar = plt.colorbar(sc, ax=ax_traj); cbar.set_label("Frame Pair")
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='purple',
               markersize=8, label='Estimated Centers'),
        Line2D([0], [0], color='black', linewidth=1.5, label='Connecting Trajectory'),
    ]
    ax_traj.legend(handles=legend_elements, loc='upper left', fontsize=11)
    plt.tight_layout()
    plt.show()

    margin = 10
    x_min_z = centers[:, 0].min() - margin; x_max_z = centers[:, 0].max() + margin
    y_min_z = centers[:, 1].min() - margin; y_max_z = centers[:, 1].max() + margin
    fig_zoom, ax_zoom = plt.subplots(figsize=(8, 8))
    sc_z = ax_zoom.scatter(centers[:, 0], centers[:, 1], c=frame_pair_indices,
                           cmap='viridis', s=80, zorder=3)
    ax_zoom.plot(centers[:, 0], centers[:, 1], color='black', linewidth=1.5, zorder=2)
    for i, (xi, yi) in enumerate(zip(centers[:, 0], centers[:, 1])):
        ax_zoom.annotate(str(frame_pair_indices[i]), xy=(xi, yi), fontsize=10,
                         ha='center', va='bottom', xytext=(0, 6), textcoords='offset points')
    ax_zoom.set_xlim(x_min_z, x_max_z)
    ax_zoom.set_ylim(y_max_z, y_min_z)
    ax_zoom.set_aspect('equal', adjustable='box')
    ax_zoom.set_xlabel("X Coordinate"); ax_zoom.set_ylabel("Y Coordinate")
    ax_zoom.set_title("Zoomed COR Trajectory")
    cbar_z = plt.colorbar(sc_z, ax=ax_zoom); cbar_z.set_label("Frame Pair")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()