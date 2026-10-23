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

import torch
import torch.nn.functional as F
from torch import nn

# -----------------------------
# User-specified paths & params
# -----------------------------
video_path = "data/ALS - Trim20s.mp4"
csv_init_angles = "data/Original_Groundtruth_Values.csv"  # initial µ_θ_r per radius
FRAME_START = 6   # inclusive starting frame index (1-based indexing assumed for user frame numbers)
FRAME_END = 24    # inclusive ending frame index
# ROI
xmin, xmax = 250, 500
ymin, ymax = 200, 450
# Radii: 5-pixel intervals up to 100
radii = [19, 17, 15, 13, 11, 9, 7, 5, 4] # Changed from tuple to list
num_radii = len(radii)
# Pyramid scales (coarse -> fine)
pyramid_scales = [0.25, 0.5, 1.0]  # Level 0: coarsest, Level 2: finest
num_levels = len(pyramid_scales)
# Basis functions
K_basis = 5  # number of orthonormal radial basis functions (Legendre up to degree K-1)
# BBVI / EM hyperparams
S_mc = 10  # MC samples
OUTER_EM_ITERS = 30
E_UPDATES = 20
M_UPDATES = 10

# Learning rates per level (coarser -> finer: larger -> smaller)
# E-step lr (for mu & logvar), M-step lr (for center and sigma_n)
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
    """
    Load frames from video_path using OpenCV.
    frame_start, frame_end are 1-based inclusive indices (to match user's numbering).
    Returns list of grayscale float32 images (H x W) normalized to [0,1].
    """
    assert os.path.exists(video_path), f"Video not found: {video_path}"
    cap = cv2.VideoCapture(video_path)
    frames = []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # OpenCV uses 0-based indexing
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
    scales: list of scale factors <1 to >0 (e.g., [0.25,0.5,1.0])
    returns list of torch tensors on device (C=1,H,W) floats
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
    Build masks for rectangular annular regions "extending outward from image center".
    Interpretation: For each radius r, define a mask for pixels with distance
    from center in [r-2.5, r+2.5) if step is 5. We'll include pixels whose radius falls within (r-2.5, r].
    We will evaluate all pixels that have radius <= r and > r-5.
    Returns list of boolean masks shaped (H,W) for each radius value in radii.
    """
    yy = np.arange(roi_h) - center[1]
    xx = np.arange(roi_w) - center[0]
    X, Y = np.meshgrid(xx, yy)
    R = np.sqrt(X**2 + Y**2)
    masks = []
    step = radii[0]  # 5
    for i, r in enumerate(radii):
        r_min = r - step
        r_max = r
        if r_min < 0:
            r_min = 0.0
        mask = (R > r_min) & (R <= r_max)
        masks.append(mask)
    return masks

def legendre_basis_on_radii(radii, max_radius, K):
    """
    Compute normalized Legendre basis functions evaluated at radii normalized to [-1,1].
    Returns Phi matrix shape (num_radii, K).
    """
    # Normalize radii to [-1,1]
    # Ensure radii is a numpy array for element-wise operations
    radii_np = np.asarray(radii, dtype=np.float32)
    normalized = 2.0 * (radii_np / float(max_radius)) - 1.0
    Phi = np.zeros((len(radii_np), K), dtype=np.float32)
    # Generate Legendre polynomials via recurrence or numpy.polynomial.legendre.legvander
    from numpy.polynomial.legendre import legvander
    V = legvander(normalized, K-1)  # shape (num_radii, K)
    # Orthonormalize columns (Gram-Schmidt) to ensure orthonormal over radii sampling
    # We'll perform QR on V to get orthonormal basis over the sampling points
    Q, R = np.linalg.qr(V)
    Phi[:] = Q[:, :K]
    return Phi  # each column is a basis function evaluated at the discrete radii

def rotation_grid_from_theta(theta_map, center, H, W):
    """
    Build a grid tensor for grid_sample that maps target image coordinates to source coordinates using rotation per-pixel.
    theta_map: torch tensor (H,W) giving rotation angles at each pixel in radians
    center: (cx, cy) in pixel coordinates (in the image coordinate system used by theta_map)
    returns grid of shape (1,H,W,2) normalized to [-1,1] for grid_sample (x,y as last dim)
    """
    ys = torch.arange(0, H, device=theta_map.device, dtype=theta_map.dtype)
    xs = torch.arange(0, W, device=theta_map.device, dtype=theta_map.dtype)
    Y, X = torch.meshgrid(ys, xs, indexing='ij')
    # convert to relative coords
    Xr = X - center[0]
    Yr = Y - center[1]
    # compute rotated coordinates: R(theta) * [Xr, Yr]
    ct = torch.cos(theta_map)
    st = torch.sin(theta_map)
    Xw = ct * Xr - st * Yr + center[0]
    Yw = st * Xr + ct * Yr + center[1]
    # normalize to [-1,1] for grid_sample
    Hf = float(H)
    Wf = float(W)
    xn = 2.0 * (Xw / (Wf - 1) - 0.5)
    yn = 2.0 * (Yw / (Hf - 1) - 0.5)
    grid = torch.stack((xn, yn), dim=-1).unsqueeze(0)  # (1,H,W,2)
    return grid

def mse_np(a, b):
    return np.mean((a - b) ** 2)

# -----------------------------
# Core Model & Training Helpers
# -----------------------------
class BBVIRotEstimator:
    def __init__(self, roi_h, roi_w, radii, basis_Phi, device):
        self.roi_h = roi_h
        self.roi_w = roi_w
        self.radii = radii
        self.K = basis_Phi.shape[1]
        self.Phi = torch.from_numpy(basis_Phi).to(device)  # (num_radii, K)
        self.device = device

    def theta_from_coeffs_per_radius(self, coeffs_w):
        """
        coeffs_w: (K,) torch coefficients for basis -> produce vector of theta at each radius (num_radii,)
        Then expand to theta_map (H,W) by assigning corresponding radius band values.
        We'll create theta_map by assigning theta for each radius region's pixels.
        """
        # compute theta at each radius value
        theta_r = self.Phi.matmul(coeffs_w)  # (num_radii,)
        return theta_r  # torch (num_radii,)

    def theta_map_from_theta_r(self, theta_r, radial_masks):
      # Infer correct size from first mask (H,W)
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
    # Load frames
    frames = load_video_frames(video_path, FRAME_START, FRAME_END)  # list of numpy HxW arrays
    num_frames = len(frames)
    # Build list of frame pairs: (i -> i+1) for i in 0..num_frames-2
    frame_pairs = [(i, i + 1) for i in range(num_frames - 1)]
    num_pairs = len(frame_pairs)

    # Crop ROI from the first frame to determine ROI dimensions and center
    frame0 = frames[0]
    roi0 = crop_roi(frame0, xmin, xmax, ymin, ymax)
    roi_h, roi_w = roi0.shape
    # Integer-valued center of ROI in ROI-local coordinates:
    roi_center_x = int(round((xmax - xmin) / 2.0))
    roi_center_y = int(round((ymax - ymin) / 2.0))
    roi_center = (roi_center_x, roi_center_y)

    # Convert radii to a numpy array for consistent numerical operations
    r_values = np.array(radii, dtype=np.float32)

    # Build radial masks (numpy) once using ROI dimensions and center
    radial_masks = make_radial_masks(roi_h, roi_w, roi_center, r_values)  # list of (H,W) boolean arrays

    max_radius = float(np.max(r_values))
    basis_Phi = legendre_basis_on_radii(r_values, max_radius, K_basis)  # (num_radii, K)

    estimator = BBVIRotEstimator(roi_h, roi_w, r_values, basis_Phi, device)

    # Load initial rotation angles CSV (assumed to have a column for radii or one row)
    if not os.path.exists(csv_init_angles):
        raise FileNotFoundError(f"Initial angles CSV not found at {csv_init_angles}")
    # init_df = pd.read_csv(csv_init_angles, header=None)
    init_df = pd.read_csv(csv_init_angles)

    # CSV contains header: Frame, Rectangle_0 ... Rectangle_19
    # So we must skip the first column ("Frame") and read columns 1..20
    if init_df.shape[1] >= num_radii + 1:
        # read first row, columns 1 through num_radii
        init_angles_arr = init_df.iloc[0, 1:num_radii+1].astype(float).values.astype(np.float32)
    else:
        # try first column with enough rows (but still skip non-numeric elements)
        possible_vals = pd.to_numeric(init_df.iloc[:, 1], errors='coerce')
        if possible_vals.notna().sum() >= num_radii:
            init_angles_arr = possible_vals.dropna().values[:num_radii].astype(np.float32)
        else:
            raise ValueError(
                "Initial angles CSV must contain at least 20 numeric rectangle values "
                "in columns 1..20 (after skipping 'Frame' column)."
            )
    # We'll keep initial per-radius mu_theta_r: users said initial rotation angles must remain fixed for first frame pair.
    # We'll interpret that as given initial mu_theta_r per radius; we will keep them fixed for processing the first frame pair.
    mu_theta_init = torch.from_numpy(init_angles_arr).to(device)  # (num_radii,) radians

    # Storage for plotting and trajectories
    all_mu_theta_r_per_pair = np.zeros((num_pairs, num_radii), dtype=np.float32)  # mean angle per radius per pair
    centers_traj = np.zeros((num_pairs, 2), dtype=np.float32)
    noise_variances_by_level = [np.zeros(OUTER_EM_ITERS, dtype=np.float32) for _ in range(num_levels)]
    mse_by_level = [np.zeros(OUTER_EM_ITERS, dtype=np.float32) for _ in range(num_levels)]
    elbo_by_level = [np.zeros(OUTER_EM_ITERS, dtype=np.float32) for _ in range(num_levels)]

    # For each frame pair we will process across pyramid levels
    sigma_n_fixed_levels = [None] * num_levels  # once computed from first pair at each level, keep fixed for subsequent pairs

    # For each pair index pidx, frames indices i->j relative to frames[] list
    for pidx, (i_f, j_f) in enumerate(frame_pairs):
        print(f"\nProcessing pair {pidx+1}/{num_pairs}: frame {FRAME_START + i_f} -> {FRAME_START + j_f}")
        # Crop ROI for both frames
        f1_full = crop_roi(frames[i_f], xmin, xmax, ymin, ymax)
        f2_full = crop_roi(frames[j_f], xmin, xmax, ymin, ymax)

        # Build pyramids for both frames
        pyr_f1 = build_image_pyramid(f1_full, pyramid_scales)  # list of (1,1,H,W) tensors
        pyr_f2 = build_image_pyramid(f2_full, pyramid_scales)

        # Initialize per-pair parameters:
        # We'll store per-level parameter sets and propagate coarse->fine initialization
        # Each level will have variational params for coefficients: mu_w (K,), logvar_w (K,)
        # M-params: sigma_n (scalar), center c (2,)
        per_level_params = []

        # If this is first pair then initialize mu from provided initial per-radius angles by projecting into coefficient space
        # We need to find basis coefficients w such that Phi @ w = mu_theta_init; solve least squares
        Phi_np = basis_Phi  # (num_radii, K)
        # Solve least squares: w_ls = (Phi^T Phi)^{-1} Phi^T mu_theta_init
        w_ls, *_ = np.linalg.lstsq(Phi_np, init_angles_arr, rcond=None)
        w_ls = np.asarray(w_ls, dtype=np.float32)[:K_basis]  # (K,)
        # initial variational params: mu = w_ls (for first pair we will keep final mu_theta fixed at radius level, but
        # at coefficient-level we must ensure that the induced mu_theta_r equals init for first pair - easiest: set mu_w = w_ls and freeze mu updates at level 0 for first pair)
        # For subsequent pairs we will use the posterior passed forward as prior.

        # default prior parameters
        prior_mu = torch.zeros(K_basis, device=device)
        prior_sigma = 1.0  # large prior variance

        # per-level init: coarse to fine
        for level_idx, scale in enumerate(pyramid_scales):
            # upsample previous level later; for now use w_ls as starting mean for the very first pair, or propagate previous pair's posterior
            # For simplicity, initialize mu_w with least-squares coefficients for first pair; for later pairs we will pass forward previous pair's posterior stored in all_mu_theta_r_per_pair
            mu_w = torch.from_numpy(w_ls).to(device).clone().float()
            logvar_w = torch.log(torch.ones_like(mu_w) * (1.0**2))  # initial small variance (1.0)
            # For subsequent pairs (pidx>0) we will reinitialize these below using previous posterior passed as prior

            # initialize sigma_n:
            sigma_n = torch.tensor(0.02, dtype=torch.float32, device=device, requires_grad=True)  # initial guess
            # center init: integer-valued ROI center in full ROI coordinates scaled to this pyramid level
            # center coordinates in ROI-local pixels scaled by scale factor
            c_x = roi_center[0] * scale
            c_y = roi_center[1] * scale
            center_param = torch.tensor([c_x, c_y], dtype=torch.float32, device=device, requires_grad=True)

            per_level_params.append({
                "mu_w": mu_w.detach().clone().requires_grad_(True),
                "logvar_w": logvar_w.detach().clone().requires_grad_(True),
                "sigma_n": sigma_n,
                "center": center_param
            })

        # If this is not the first pair, initialize using previous pair's posterior (posterior -> prior)
        if pidx > 0:
            # We have previous pair's radii means all_mu_theta_r_per_pair[pidx-1, :]
            prev_mu_theta = torch.from_numpy(all_mu_theta_r_per_pair[pidx-1, :]).to(device)  # (num_radii,)
            # compute least squares to get coefficients
            prev_w_ls, *_ = np.linalg.lstsq(Phi_np, prev_mu_theta.cpu().numpy(), rcond=None)
            prev_w_ls = torch.from_numpy(np.asarray(prev_w_ls, dtype=np.float32)[:K_basis]).to(device)
            for level_idx in range(num_levels):
                per_level_params[level_idx]["mu_w"] = prev_w_ls.clone().detach().requires_grad_(True)
                per_level_params[level_idx]["logvar_w"] = torch.log(torch.ones_like(prev_w_ls) * (1.0**2)).requires_grad_(True)
                # keep center initialized to ROI center scaled
                c_x = roi_center[0] * pyramid_scales[level_idx]
                c_y = roi_center[1] * pyramid_scales[level_idx]
                per_level_params[level_idx]["center"] = torch.tensor([c_x, c_y], dtype=torch.float32, device=device, requires_grad=True)
                per_level_params[level_idx]["sigma_n"] = torch.tensor(0.02, dtype=torch.float32, device=device, requires_grad=True)

        # Now iterate over pyramid levels coarse->fine
        for level_idx, scale in enumerate(pyramid_scales):
            print(f"  Level {level_idx} scale {scale}")
            # get images at this level (ROI-local)
            f1_t = pyr_f1[level_idx].to(device)  # (1,1,H,W)
            f2_t = pyr_f2[level_idx].to(device)

            Hlvl = f1_t.shape[2]
            Wlvl = f1_t.shape[3]

            # Scale radial masks to current level size if needed (nearest-resize to match dims)
            if scale != 1.0:
                # Resize masks to Hlvl x Wlvl using nearest interpolation
                scaled_masks = []
                for mask in radial_masks:
                    mask_uint8 = (mask.astype(np.uint8) * 255)
                    mask_rs = cv2.resize(mask_uint8, (Wlvl, Hlvl), interpolation=cv2.INTER_NEAREST)
                    scaled_masks.append(mask_rs.astype(bool))
            else:
                scaled_masks = radial_masks

            # Build estimator using same basis (Phi is defined on original radii sampling; in coarse levels we still use same radii discretization
            # but center coordinates scale changes. We will use the same Phi and map theta_r values to radius bands at the scaled grid.)
            # Prepare parameters for optimization
            params = per_level_params[level_idx]

            # Freeze mu_theta for first pair as required: if pidx==0 (first pair), initial rotation angles must remain fixed for first pair.
            # We'll implement this by not updating the coefficient mean mu_w (i.e., set grad off) during E-step updates for the first pair and first outer EM iterations.
            freeze_mu_for_first_pair = (pidx == 0)

            # Create optimizers: separate for E-step and M-step as requested.
            # E-step optimizes variational mu_w and logvar_w
            mu_w = params["mu_w"]
            logvar_w = params["logvar_w"]
            sigma_n = params["sigma_n"]
            center_param = params["center"]

            # Ensure parameters require grad
            mu_w = mu_w.requires_grad_(True)
            logvar_w = logvar_w.requires_grad_(True)
            sigma_n = sigma_n.requires_grad_(True)
            center_param = center_param.requires_grad_(True)

            # Replace in params
            params["mu_w"] = mu_w
            params["logvar_w"] = logvar_w
            params["sigma_n"] = sigma_n
            params["center"] = center_param

            optimizerE = torch.optim.Adam([mu_w, logvar_w], lr=Elr_levels[level_idx])
            optimizerM = torch.optim.Adam([center_param, sigma_n], lr=Mlr_levels[level_idx])

            # If sigma_n already computed for this level from first pair and we are after first pair, set sigma_n fixed
            if (sigma_n_fixed_levels[level_idx] is not None) and (pidx > 0):
                # load fixed value and freeze sigma_n param by overriding gradient updates in M-step (we'll keep it constant)
                fixed_sigma_val = float(sigma_n_fixed_levels[level_idx])
                sigma_n = torch.tensor(fixed_sigma_val, dtype=torch.float32, device=device, requires_grad=False)
                params["sigma_n"] = sigma_n
                # Re-init optimizer M without sigma_n param
                optimizerM = torch.optim.Adam([center_param], lr=Mlr_levels[level_idx])

            # Outer EM loop
            for outer in range(OUTER_EM_ITERS):
                # E-step: update variational params mu_w, logvar_w for E_UPDATES steps
                for e_it in range(E_UPDATES):
                    optimizerE.zero_grad()
                    # Sample S_mc epsilons
                    std_w = torch.exp(0.5 * logvar_w)
                    # sample shape (S_mc, K)
                    eps = torch.randn((S_mc, mu_w.shape[0]), device=device, dtype=mu_w.dtype)
                    w_s = mu_w.unsqueeze(0) + std_w.unsqueeze(0) * eps  # (S, K)
                    # For first pair and freeze flag: don't compute gradients wrt mu_w (keep mu fixed)
                    # We'll compute loss and backward; if freeze_mu_for_first_pair, temporarily disable gradient for mu_w by detaching mu_w in w_s creation
                    if freeze_mu_for_first_pair:
                        # keep mu fixed: zero grad allowed on mu by creating w_s that depends on detached mu
                        mu_w_det = mu_w.detach()
                        w_s = mu_w_det.unsqueeze(0) + std_w.unsqueeze(0) * eps
                    # Evaluate log-likelihood term per sample
                    losses = []
                    elbo_samples = []
                    kl_term = 0.0
                    for s in range(S_mc):
                        w_sample = w_s[s]  # (K,)
                        # compute theta_r (num_radii,)
                        theta_r = estimator.theta_from_coeffs_per_radius(w_sample)  # (num_radii,)
                        # build theta_map (Hlvl, Wlvl)
                        theta_map = estimator.theta_map_from_theta_r(theta_r, scaled_masks)  # (H,W) torch
                        # rotate f2 by theta_map around center_param
                        grid = rotation_grid_from_theta(theta_map, (center_param[0], center_param[1]), Hlvl, Wlvl)
                        # sample
                        f2_warp = F.grid_sample(f2_t, grid, mode='bilinear', padding_mode='border', align_corners=True)
                        # compute sum squared error
                        res = f1_t - f2_warp
                        se = (res ** 2).sum()  # scalar
                        losses.append(se)
                    # Monte Carlo expected negative log-likelihood
                    mc_se = torch.stack(losses).mean()
                    # ELBO: E_q[log p(x|w)] - KL(q||p)
                    # Likelihood for Gaussian noise: log p(x|w) = -1/(2 sigma_n^2) * SE + const
                    if isinstance(params["sigma_n"], torch.Tensor) and params["sigma_n"].requires_grad:
                        sigma_n_param = params["sigma_n"]
                    else:
                        sigma_n_param = params["sigma_n"]
                    loglik = -0.5 * mc_se / (sigma_n_param ** 2)
                    # KL between q(w) = N(mu, diag(var)) and prior N(0, prior_sigma^2)
                    var_diag = torch.exp(logvar_w)
                    prior_var = prior_sigma ** 2
                    kl = 0.5 * torch.sum((var_diag + mu_w ** 2) / prior_var - 1.0 - torch.log(var_diag) + math.log(prior_var))
                    elbo = loglik - kl
                    loss = -elbo  # maximize elbo -> minimize -elbo
                    loss.backward()
                    # If freeze mu for first pair, zero its grad so mu_w won't update
                    if freeze_mu_for_first_pair:
                        if mu_w.grad is not None:
                            mu_w.grad.zero_()
                    optimizerE.step()
                # End E-step updates

                # M-step: update sigma_n and center (M_UPDATES steps)
                for m_it in range(M_UPDATES):
                    optimizerM.zero_grad()
                    # Evaluate expected squared error under q(w) (MC)
                    with torch.set_grad_enabled(True):
                        std_w = torch.exp(0.5 * logvar_w)
                        eps = torch.randn((S_mc, mu_w.shape[0]), device=device, dtype=mu_w.dtype)
                        w_s = mu_w.unsqueeze(0) + std_w.unsqueeze(0) * eps
                        losses_m = []
                        elbo_samples_m = []
                        for s in range(S_mc):
                            w_sample = w_s[s]
                            theta_r = estimator.theta_from_coeffs_per_radius(w_sample)
                            theta_map = estimator.theta_map_from_theta_r(theta_r, scaled_masks)
                            grid = rotation_grid_from_theta(theta_map, (center_param[0], center_param[1]), Hlvl, Wlvl)
                            f2_warp = F.grid_sample(f2_t, grid, mode='bilinear', padding_mode='border', align_corners=True)
                            res = f1_t - f2_warp
                            se = (res ** 2).sum()
                            losses_m.append(se)
                        mc_se_m = torch.stack(losses_m).mean()
                        # negative ELBO with current q approximated (we treat q fixed here)
                        loglik_m = -0.5 * mc_se_m / (sigma_n_param ** 2)
                        # KL same as above (depends on mu_w and logvar_w)
                        var_diag = torch.exp(logvar_w)
                        kl = 0.5 * torch.sum((var_diag + mu_w ** 2) / prior_var - 1.0 - torch.log(var_diag) + math.log(prior_var))
                        elbo_m = loglik_m - kl
                        loss_m = -elbo_m
                        loss_m.backward()
                        optimizerM.step()

                # End M-step updates

                # After finishing this outer iteration record metrics: sigma_n (scalar), MSE (mc_se / num_pixels), ELBO value
                # Compute numeric MSE and ELBO for logging
                with torch.no_grad():
                    # compute current expected SE via MC
                    std_w = torch.exp(0.5 * logvar_w)
                    eps = torch.randn((S_mc, mu_w.shape[0]), device=device, dtype=mu_w.dtype)
                    w_s = mu_w.unsqueeze(0) + std_w.unsqueeze(0) * eps
                    losses_log = []
                    for s in range(S_mc):
                        w_sample = w_s[s]
                        theta_r = estimator.theta_from_coeffs_per_radius(w_sample)
                        theta_map = estimator.theta_map_from_theta_r(theta_r, scaled_masks)
                        grid = rotation_grid_from_theta(theta_map, (center_param[0], center_param[1]), Hlvl, Wlvl)
                        f2_warp = F.grid_sample(f2_t, grid, mode='bilinear', padding_mode='border', align_corners=True)
                        res = f1_t - f2_warp
                        se = (res ** 2).sum()
                        losses_log.append(se)
                    mc_se_log = float(torch.stack(losses_log).mean().cpu().numpy())
                    num_pixels = float(Hlvl * Wlvl)
                    mse_val = mc_se_log / num_pixels
                    # ELBO estimate:
                    if isinstance(params["sigma_n"], torch.Tensor) and not params["sigma_n"].requires_grad:
                        sigma_n_val = params["sigma_n"].item()
                    else:
                        sigma_n_val = float(sigma_n.detach().cpu().numpy())
                    loglik_val = -0.5 * mc_se_log / (sigma_n_val ** 2)
                    var_diag_np = float(torch.sum(torch.exp(logvar_w)).detach().cpu().numpy())
                    mu_w_np = float(torch.sum(mu_w ** 2).detach().cpu().numpy())
                    # approximate KL (same as above)
                    kl_val = 0.5 * (var_diag_np + mu_w_np) / prior_var - 0.5 * K_basis - 0.5 * float(torch.sum(logvar_w).detach().cpu().numpy()) + 0.5 * K_basis * math.log(prior_var)
                    # Note: above scalar computation is approximate aggregated scalar; we instead compute exact:
                    var_diag = torch.exp(logvar_w)
                    kl_exact = 0.5 * torch.sum((var_diag + mu_w ** 2) / prior_var - 1.0 - torch.log(var_diag) + math.log(prior_var)).item()
                    elbo_val = loglik_val - kl_exact

                    # Save metrics for this level, this outer iteration (index outer)
                    noise_variances_by_level[level_idx][outer] = sigma_n_val
                    mse_by_level[level_idx][outer] = mse_val
                    elbo_by_level[level_idx][outer] = elbo_val

                # print progress for the level for every 5 iterations
                if (outer + 1) % 5 == 0:
                    print(f"    outer {outer+1}/{OUTER_EM_ITERS}: sigma_n={sigma_n_val:.6f}, mse={mse_val:.6e}, elbo={elbo_val:.4f}")

            # End outer loop for this level

            # After finishing level, store final params back
            per_level_params[level_idx]["mu_w"] = mu_w.detach()
            per_level_params[level_idx]["logvar_w"] = logvar_w.detach()
            per_level_params[level_idx]["sigma_n"] = sigma_n.detach() if isinstance(sigma_n, torch.Tensor) else sigma_n
            per_level_params[level_idx]["center"] = center_param.detach()

            # For first pair, fix sigma_n for subsequent pairs at this level (compute from this pair)
            if pidx == 0:
                sigma_n_fixed_levels[level_idx] = float(per_level_params[level_idx]["sigma_n"].cpu().numpy())

            # If not the finest level, upsample coefficient means to initialize next finer level.
            # We upsample coefficients by solving least squares to match mu_theta_r at this level then reuse for next level initialization.
            if level_idx < num_levels - 1:
                # compute final mu_theta_r from mu_w via Phi @ mu_w
                mu_w_final = per_level_params[level_idx]["mu_w"].cpu().numpy()
                mu_theta_r_final = basis_Phi.dot(mu_w_final)  # (num_radii,)
                # For next level initialization, compute least squares again to get coefficients (should be close)
                w_next, *_ = np.linalg.lstsq(Phi_np, mu_theta_r_final, rcond=None)
                w_next = np.asarray(w_next, dtype=np.float32)[:K_basis]
                per_level_params[level_idx + 1]["mu_w"] = torch.from_numpy(w_next).to(device).requires_grad_(True)
                per_level_params[level_idx + 1]["logvar_w"] = per_level_params[level_idx]["logvar_w"].clone().detach().requires_grad_(True)
                # propagate center upscaled to next level (we assume coordinates scale-up)
                cx = per_level_params[level_idx]["center"][0].cpu().numpy() * (pyramid_scales[level_idx + 1] / pyramid_scales[level_idx])
                cy = per_level_params[level_idx]["center"][1].cpu().numpy() * (pyramid_scales[level_idx + 1] / pyramid_scales[level_idx])
                per_level_params[level_idx + 1]["center"] = torch.tensor([cx, cy], dtype=torch.float32, device=device, requires_grad=True)
                per_level_params[level_idx + 1]["sigma_n"] = per_level_params[level_idx]["sigma_n"]
        # End pyramid levels for this frame pair

        # After finishing all levels for this pair, record final mu_theta_r (from finest level)
        mu_w_fin = per_level_params[-1]["mu_w"].cpu().numpy()
        mu_theta_fin = basis_Phi.dot(mu_w_fin)  # (num_radii,)
        all_mu_theta_r_per_pair[pidx, :] = mu_theta_fin.copy()
        centers_traj[pidx, :] = per_level_params[-1]["center"].cpu().numpy()
        print(f"  Finished pair {pidx+1}: center={centers_traj[pidx, :]}, mean_theta_r_sample[:5]={mu_theta_fin[:5]}")

    # End all pairs processing

    # -----------------------------
    # Plotting section
    # -----------------------------
    # Rotation Angle vs Radius plot: show µ_θ_r * radius across frames for each radius value
    # frames_indices = np.arange(FRAME_START, FRAME_START + num_frames - 0)  # frame numbers corresponding to frames list
    # But frame pairs correspond to FRAME_START..FRAME_END-1 (start frames for pairs)
    frame_pair_indices = np.arange(FRAME_START, FRAME_START + num_pairs)
    # Build figure
    fig1, ax1 = plt.subplots(figsize=(12, 7))
    # Plot one curve per radius, y = mu_theta_r * radius
    for ridx, r in enumerate(r_values): # Use r_values for plotting
        yvals = all_mu_theta_r_per_pair[:, ridx] * float(r)  # scaled by radius
        # yvals = all_mu_theta_r_per_pair[:, ridx]  # scaled by radius
        ax1.plot(frame_pair_indices, yvals, label=f"r={int(r)}", marker='o', linewidth=2, markersize=6)

    ax1.set_xticks([6, 7.5, 10.0, 12.5, 15.0, 17.5, 20.0, 22.5, 24])
    ax1.set_xlabel("Frame index")
    ax1.set_ylabel("Rotation Values (scaled by radius)")
    ax1.set_title("Rotation Values (µ_θ_r scaled by radius)")
    # only show legend entries for some radii to avoid clutter
    ax1.legend(loc='upper right', ncol=1, fontsize='small')
    plt.axhline(0, color='black', linestyle='--', linewidth=1)
    plt.tight_layout()

    # Noise Variance, MSE, ELBO plots across pyramid levels
    metrics = [
        ("Noise Variance", noise_variances_by_level),
        ("MSE", mse_by_level),
        ("ELBO", elbo_by_level)
    ]

    for metric_name, metric_data in metrics:

        # We need 3 rows on the right side (one per level) and 1 column for main plot
        fig = plt.figure(figsize=(14, 7))

        # MAIN PLOT (left side, rows 0–2, col 0)
        ax_main = plt.subplot2grid((3, 2), (0, 0), rowspan=3, colspan=1)
        colors = ['C0', 'C1', 'C2']

        # Combined main curves
        for lvl in range(num_levels):
            ax_main.plot(
                np.arange(1, OUTER_EM_ITERS + 1),
                metric_data[lvl],
                color=colors[lvl],
                label=f"Level {lvl}"
            )

        ax_main.set_xlabel("Overall Iterations")
        ax_main.set_ylabel(metric_name)
        ax_main.set_title(f"{metric_name} Across Pyramid Levels")
        ax_main.legend()

        # THREE SMALL SUBPLOTS (right column, 3 rows)
        for lvl in range(num_levels):
            ax_small = plt.subplot2grid((3, 2), (lvl, 1))
            ax_small.plot(
                np.arange(1, OUTER_EM_ITERS + 1),
                metric_data[lvl],
                color=colors[lvl]
            )
            ax_small.set_title(f"Level {lvl} ({metric_name})")
            ax_small.set_xlabel("Overall Iterations")
            ax_small.set_ylabel(metric_name)

        plt.tight_layout()
        plt.show()


    # Plot center trajectory on top of a reference (first ROI frame)
    ref_img = frames[0][::]  # full frame
    # reconstruct full-frame-level plotting for ROI overlay: show ROI region crop for display
    frame_rgb = cv2.cvtColor((ref_img * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
    # create figure for trajectory (full ROI)
    fig_traj, ax_traj = plt.subplots(figsize=(8, 8))
    ax_traj.imshow(crop_roi(ref_img, xmin, xmax, ymin, ymax), cmap='gray', origin='upper')
    # plot ROI rectangle over the displayed ROI (it's the displayed area)
    # centers_traj are ROI-local coordinates relative to ROI top-left; they were recorded at finest level scale
    # However during plotting we want them in ROI pixel coords corresponding to the displayed ROI (full scale)
    # Our recorded centers are in final level coordinates (scaled according to pyramid scale 1.0) so they are directly ROI-local
    sc = ax_traj.scatter(centers_traj[:, 0], centers_traj[:, 1], c=np.arange(1, num_pairs + 1), cmap='viridis', s=40)
    # connect by line
    ax_traj.plot(centers_traj[:, 0], centers_traj[:, 1], color='white', linewidth=1.0, label='Center Trajectory')
    # ROI drawn as red dashed rectangle (the display itself is ROI, so rectangle is full bounds)
    ax_traj.add_patch(plt.Rectangle((0, 0), roi_w - 1, roi_h - 1, fill=False, edgecolor='red', linestyle='--', linewidth=2, label='ROI'))
    cbar = fig_traj.colorbar(sc, ax=ax_traj)
    cbar.set_label('Frame Pair')
    ax_traj.set_title("Complete Trajectory of Estimated Rotation Centers (ROI)")
    ax_traj.legend()
    plt.tight_layout()

    # Zoomed-in trajectory with annotations
    fig_zoom, ax_zoom = plt.subplots(figsize=(6, 6))
    ax_zoom.imshow(crop_roi(ref_img, xmin, xmax, ymin, ymax), cmap='gray', origin='upper')
    ax_zoom.plot(centers_traj[:, 0], centers_traj[:, 1], marker='o', linewidth=1.2, label='Center Trajectory')
    for idx, (cx, cy) in enumerate(centers_traj):
        ax_zoom.annotate(str(FRAME_START + idx), (cx, cy), color='yellow', fontsize=8)
    sc2 = ax_zoom.scatter(centers_traj[:, 0], centers_traj[:, 1], c=np.arange(1, num_pairs + 1), cmap='viridis', s=40)
    cbar2 = fig_zoom.colorbar(sc2, ax=ax_zoom)
    cbar2.set_label('Frame index')
    ax_zoom.set_title('Zoomed Center of Rotation Trajectory with Adaptive Center within Same ROI')
    ax_zoom.legend()
    plt.tight_layout()
    plt.show()

    # Show Rotation Angle vs Radius figure
    plt.show(fig1)

if __name__ == "__main__":
    main()