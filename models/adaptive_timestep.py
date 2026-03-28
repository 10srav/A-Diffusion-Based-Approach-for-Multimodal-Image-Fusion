"""
Content-Adaptive Timestep Selection for IR-VIS Diffusion Fusion

Non-learned CV-based module that analyzes each IR+VIS pair and selects
a per-image diffusion timestep based on actual image content.

Key insight: cross-modal metrics (how different IR and VIS are from each
other) are far more discriminative per-pair than single-modal metrics.
A forest-at-night pair vs. a city-in-daylight pair will have very different
structural divergence, histogram distance, and cross-correlation.
"""
import torch
import numpy as np
import cv2


class AdaptiveTimestep:
    """
    Selects per-image diffusion timestep from IR+VIS content analysis.

    Produces genuinely different timesteps for different image pairs by
    combining cross-modal metrics (50% weight) with per-modal complexity
    metrics (50% weight). Cross-modal metrics capture how much the IR
    and VIS modalities disagree, which is the dominant factor in fusion
    difficulty.

    Args:
        T: Maximum timestep (number of diffusion steps)
        device: Torch device (used only for input tensor conversion)
    """

    def __init__(self, T=50, min_t=None, max_t=None, device="cuda"):
        self.T = T
        self.min_t = min_t if min_t is not None else 0
        self.max_t = max_t if max_t is not None else T - 1
        self.device = device

    def _to_gray(self, tensor):
        """Convert [B, C, H, W] tensor in [-1,1] or [0,1] to grayscale uint8 numpy."""
        img = tensor.detach().cpu().squeeze(0)  # [C, H, W]
        if img.ndim == 3 and img.shape[0] == 3:
            gray = 0.299 * img[0] + 0.587 * img[1] + 0.114 * img[2]
        elif img.ndim == 3 and img.shape[0] == 1:
            gray = img.squeeze(0)
        else:
            gray = img
        # Handle [-1, 1] range
        if gray.min() < -0.1:
            gray = (gray + 1.0) / 2.0
        gray = (gray.numpy() * 255).clip(0, 255).astype(np.uint8)
        return gray

    # ------------------------------------------------------------------
    # Cross-modal metrics (these differentiate pairs the most)
    # ------------------------------------------------------------------

    def _gradient_map_ncc(self, gray_vis, gray_ir):
        """
        Normalized cross-correlation of gradient magnitude maps.
        Returns 1 - NCC so 0=identical structure, 1=completely different.
        Uses Sobel gradient magnitudes blurred slightly for robustness.
        """
        def grad_mag(g):
            gx = cv2.Sobel(g, cv2.CV_64F, 1, 0, ksize=3)
            gy = cv2.Sobel(g, cv2.CV_64F, 0, 1, ksize=3)
            mag = np.sqrt(gx ** 2 + gy ** 2)
            return cv2.GaussianBlur(mag, (5, 5), 1.0)

        mag_vis = grad_mag(gray_vis).flatten()
        mag_ir = grad_mag(gray_ir).flatten()
        std_v, std_r = np.std(mag_vis), np.std(mag_ir)
        if std_v < 1e-6 or std_r < 1e-6:
            return 1.0
        ncc = np.corrcoef(mag_vis, mag_ir)[0, 1]
        return float(np.clip(1.0 - ncc, 0, 1))

    def _histogram_distance(self, gray_vis, gray_ir):
        """
        Earth Mover's Distance between VIS and IR intensity histograms.
        Normalized to [0, 1]. Higher = more different distributions.
        """
        hist_vis = cv2.calcHist([gray_vis], [0], None, [256], [0, 256]).flatten()
        hist_ir = cv2.calcHist([gray_ir], [0], None, [256], [0, 256]).flatten()
        hist_vis = hist_vis / (hist_vis.sum() + 1e-8)
        hist_ir = hist_ir / (hist_ir.sum() + 1e-8)
        # CDF-based EMD (Wasserstein-1 for 1D histograms)
        cdf_vis = np.cumsum(hist_vis)
        cdf_ir = np.cumsum(hist_ir)
        emd = np.sum(np.abs(cdf_vis - cdf_ir)) / 256.0
        return float(emd)

    def _cross_correlation(self, gray_vis, gray_ir):
        """
        Pearson correlation between VIS and IR.
        Returns 1 - |corr| so low correlation (harder fusion) = higher score.
        """
        v = gray_vis.astype(np.float64).flatten()
        r = gray_ir.astype(np.float64).flatten()
        std_v, std_r = np.std(v), np.std(r)
        if std_v < 1.0 or std_r < 1.0:
            return 1.0  # constant image = maximally difficult
        corr = np.corrcoef(v, r)[0, 1]
        return float(1.0 - abs(corr))

    def _gradient_angle_divergence(self, gray_vis, gray_ir):
        """
        Mean angular difference between VIS and IR gradient directions.
        Captures structural orientation disagreement.
        """
        def grad_angles(g):
            gx = cv2.Sobel(g, cv2.CV_64F, 1, 0, ksize=3)
            gy = cv2.Sobel(g, cv2.CV_64F, 0, 1, ksize=3)
            return np.arctan2(gy, gx)

        angles_vis = grad_angles(gray_vis)
        angles_ir = grad_angles(gray_ir)
        # Angular difference, wrapped to [0, pi]
        diff = np.abs(angles_vis - angles_ir)
        diff = np.minimum(diff, 2 * np.pi - diff)
        # Weight by gradient magnitude so flat areas don't dominate
        mag_vis = cv2.Sobel(gray_vis, cv2.CV_64F, 1, 0, ksize=3) ** 2 + \
                  cv2.Sobel(gray_vis, cv2.CV_64F, 0, 1, ksize=3) ** 2
        mag_ir = cv2.Sobel(gray_ir, cv2.CV_64F, 1, 0, ksize=3) ** 2 + \
                 cv2.Sobel(gray_ir, cv2.CV_64F, 0, 1, ksize=3) ** 2
        weight = np.sqrt(mag_vis) + np.sqrt(mag_ir) + 1e-8
        weighted_diff = np.sum(diff * weight) / np.sum(weight)
        return float(weighted_diff / np.pi)  # normalize to [0, 1]

    # ------------------------------------------------------------------
    # Per-modal complexity metrics
    # ------------------------------------------------------------------

    def _gradient_complexity(self, gray):
        """
        Coefficient of variation of gradient magnitudes.
        High variation = mixed textures (complex), low = uniform.
        """
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        mag = np.sqrt(gx ** 2 + gy ** 2)
        mean_mag = np.mean(mag)
        if mean_mag < 1.0:
            return 0.0
        return float(np.std(mag) / mean_mag)

    def _detail_energy(self, gray):
        """
        Normalized Laplacian energy - robust high-frequency content measure.
        Higher = more fine detail. Uses log scale for better dynamic range.
        """
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        energy = np.mean(lap ** 2)
        # Log scale maps typical range [10, 5000] to [0, 1]
        return float(np.clip(np.log1p(energy) / np.log1p(3000), 0, 1))

    def _local_contrast_variance(self, gray, block_size=32):
        """
        Variance of per-block standard deviations.
        Images with spatially varying contrast (e.g. bright sky + dark ground)
        score higher than uniformly textured images.
        """
        h, w = gray.shape
        blocks = []
        for i in range(0, h - block_size + 1, block_size):
            for j in range(0, w - block_size + 1, block_size):
                block = gray[i:i + block_size, j:j + block_size].astype(np.float64)
                blocks.append(np.std(block))
        if len(blocks) < 2:
            return 0.0
        blocks = np.array(blocks)
        mean_c = np.mean(blocks)
        if mean_c < 1.0:
            return 0.0
        return float(np.std(blocks) / mean_c)

    # ------------------------------------------------------------------
    # Main interface
    # ------------------------------------------------------------------

    def get_timestep(self, vis, ir, return_details=False):
        """
        Compute a content-adaptive diffusion timestep for this image pair.

        Args:
            vis: [1, C, H, W] visible image tensor
            ir:  [1, C, H, W] infrared image tensor
            return_details: if True, also return per-metric breakdown

        Returns:
            t: int, selected timestep in [min_t, max_t] (default [0, T-1])
            details: dict (only if return_details=True) with all metric
                     values and the composite score
        """
        vis_np = self._to_gray(vis)
        ir_np = self._to_gray(ir)

        # --- Cross-modal metrics (most discriminative per-pair) ---
        grad_ncc = self._gradient_map_ncc(vis_np, ir_np)
        hist_dist = self._histogram_distance(vis_np, ir_np)
        cross_corr = self._cross_correlation(vis_np, ir_np)
        angle_div = self._gradient_angle_divergence(vis_np, ir_np)

        # --- Per-modal complexity ---
        vis_grad = self._gradient_complexity(vis_np)
        ir_grad = self._gradient_complexity(ir_np)
        vis_detail = self._detail_energy(vis_np)
        ir_detail = self._detail_energy(ir_np)
        vis_contrast = self._local_contrast_variance(vis_np)
        ir_contrast = self._local_contrast_variance(ir_np)

        # --- Normalize per-modal metrics to [0, 1] ---
        grad_score = np.clip((vis_grad + ir_grad) / 2.0 / 2.5, 0, 1)
        detail_score = (vis_detail + ir_detail) / 2.0  # already [0, 1]
        contrast_score = np.clip((vis_contrast + ir_contrast) / 2.0 / 1.5, 0, 1)

        # --- Composite score ---
        # Cross-modal (55%): these drive most of the per-pair variation
        # Per-modal (45%): overall scene complexity
        raw_score = (
            # Cross-modal (55%)
            0.15 * cross_corr +       # widest range observed
            0.15 * hist_dist * 3.0 +   # amplify (raw range ~0.07-0.26)
            0.13 * grad_ncc +
            0.12 * angle_div +
            # Per-modal (45%)
            0.15 * grad_score +
            0.15 * detail_score +
            0.15 * contrast_score
        )

        # --- Score stretching ---
        # Raw scores for same-domain datasets (e.g. M3FD driving scenes)
        # cluster in a narrow band. Apply a sigmoid-like stretch centered
        # on the empirical midpoint to spread them across the full range.
        # sigmoid(x) = 1 / (1 + exp(-gain * (x - center)))
        center = 0.65
        gain = 18.0
        stretched = 1.0 / (1.0 + np.exp(-gain * (raw_score - center)))

        score = float(np.clip(stretched, 0, 1))

        # Map to timestep (full range, no artificial clamping)
        t = int(score * (self.T - 1))
        t = max(self.min_t, min(self.max_t, t))

        if return_details:
            details = {
                'raw_score': float(raw_score),
                'score': score,
                'timestep': t,
                # Cross-modal
                'gradient_map_ncc': grad_ncc,
                'histogram_distance': hist_dist,
                'cross_correlation_inv': cross_corr,
                'gradient_angle_divergence': angle_div,
                # Per-modal (raw)
                'vis_gradient_complexity': vis_grad,
                'ir_gradient_complexity': ir_grad,
                'vis_detail_energy': vis_detail,
                'ir_detail_energy': ir_detail,
                'vis_local_contrast_var': vis_contrast,
                'ir_local_contrast_var': ir_contrast,
                # Per-modal (normalized)
                'gradient_score': float(grad_score),
                'detail_score': float(detail_score),
                'contrast_score': float(contrast_score),
            }
            return t, details

        return t
