"""
Content Analyzer for Adaptive Diffusion
Analyzes IR+VIS image pairs to produce content complexity descriptors
that drive adaptive noise schedules, timestep selection, and sampling.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .adaptive_unet import _num_groups


class ContentAnalyzer(nn.Module):
    """
    Lightweight CNN that analyzes IR+VIS image pairs to extract
    content complexity descriptors (~0.5M params).

    Outputs:
        content_descriptor: [B, descriptor_dim] global complexity vector
        complexity_map:     [B, 1, 128, 128] spatial complexity heatmap
        complexity_scalar:  [B, 1] overall complexity in [0, 1]
    """

    def __init__(self, descriptor_dim=64):
        super().__init__()
        self.descriptor_dim = descriptor_dim

        # Global feature extraction from concatenated IR+VIS (6 channels)
        self.stem = nn.Sequential(
            nn.Conv2d(6, 32, 7, stride=4, padding=3),    # 256->64
            nn.GroupNorm(_num_groups(32), 32),
            nn.SiLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),   # 64->32
            nn.GroupNorm(_num_groups(64), 64),
            nn.SiLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),  # 32->16
            nn.GroupNorm(_num_groups(128), 128),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),                      # 16->1x1
        )
        self.global_head = nn.Sequential(
            nn.Linear(128, descriptor_dim),
            nn.SiLU(),
        )

        # Spatial complexity map (per-region analysis)
        self.spatial_stem = nn.Sequential(
            nn.Conv2d(6, 32, 3, stride=2, padding=1),    # 256->128
            nn.GroupNorm(_num_groups(32), 32),
            nn.SiLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.GroupNorm(_num_groups(32), 32),
            nn.SiLU(),
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, ir_image, vis_image):
        """
        Args:
            ir_image:  [B, 3, 256, 256] in [-1, 1]
            vis_image: [B, 3, 256, 256] in [-1, 1]

        Returns:
            content_descriptor: [B, descriptor_dim] global complexity vector
            complexity_map:     [B, 1, 128, 128] spatial complexity in [0, 1]
            complexity_scalar:  [B, 1] mean complexity in [0, 1]
        """
        x = torch.cat([ir_image, vis_image], dim=1)  # [B, 6, 256, 256]

        # Global descriptor
        global_feat = self.stem(x).squeeze(-1).squeeze(-1)  # [B, 128]
        content_descriptor = self.global_head(global_feat)   # [B, descriptor_dim]

        # Spatial complexity
        complexity_map = self.spatial_stem(x)                # [B, 1, 128, 128]
        complexity_scalar = complexity_map.mean(dim=[2, 3])  # [B, 1]

        return content_descriptor, complexity_map, complexity_scalar


class AdaptiveTimestepSelector(nn.Module):
    """
    Predicts per-image DDIM step count and non-uniform timestep density
    from content descriptors (~0.1M params).

    Complex scenes get more steps concentrated at mid-range timesteps
    where structural details emerge. Simple scenes get fewer steps.
    """

    def __init__(self, descriptor_dim=64, max_steps=50,
                 min_steps=15, num_anchor_bins=50):
        super().__init__()
        self.max_steps = max_steps
        self.min_steps = min_steps
        self.num_anchor_bins = num_anchor_bins

        # Predict step count from complexity
        self.step_predictor = nn.Sequential(
            nn.Linear(descriptor_dim, 32),
            nn.SiLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),  # output in [0, 1], scaled to [min_steps, max_steps]
        )

        # Predict timestep density: where to concentrate steps
        self.density_predictor = nn.Sequential(
            nn.Linear(descriptor_dim, 64),
            nn.SiLU(),
            nn.Linear(64, num_anchor_bins),
        )

    def forward(self, content_descriptor):
        """
        Args:
            content_descriptor: [B, descriptor_dim]

        Returns:
            num_steps: [B] integer step counts in [min_steps, max_steps]
            timestep_density: [B, num_anchor_bins] softmax density
        """
        step_frac = self.step_predictor(content_descriptor).squeeze(-1)  # [B]
        num_steps = (self.min_steps +
                     step_frac * (self.max_steps - self.min_steps))
        num_steps = num_steps.round().long().clamp(self.min_steps, self.max_steps)

        density_logits = self.density_predictor(content_descriptor)
        timestep_density = F.softmax(density_logits, dim=-1)  # [B, num_anchor_bins]

        return num_steps, timestep_density

    def build_schedule(self, num_steps_i, density_i, T=1000):
        """
        Build a non-uniform DDIM timestep schedule for a single image
        using inverse-CDF sampling from the predicted density.

        Args:
            num_steps_i: int, number of steps for this image
            density_i: [num_anchor_bins] density weights (on any device)
            T: total diffusion timesteps

        Returns:
            timesteps: sorted list of ints (ascending), length ~num_steps_i
        """
        density_i = density_i.detach().float()

        # Build CDF from density
        cdf = torch.cumsum(density_i, dim=0)
        cdf = cdf / cdf[-1]  # normalize to [0, 1]

        # Sample num_steps_i evenly spaced points in [0, 1]
        uniform_points = torch.linspace(
            0, 1, num_steps_i, device=density_i.device
        )

        # Map through inverse CDF to get non-uniform positions
        indices = torch.searchsorted(cdf, uniform_points)
        indices = indices.clamp(0, len(cdf) - 1)

        # Map anchor indices to actual timesteps in [0, T-1]
        timesteps = (indices.float() /
                     max(self.num_anchor_bins - 1, 1) * (T - 1)).long()

        # Deduplicate and sort ascending
        timesteps = timesteps.unique().sort().values.tolist()

        # Ensure 0 and T-1 are included
        if 0 not in timesteps:
            timesteps = [0] + timesteps
        if (T - 1) not in timesteps:
            timesteps.append(T - 1)

        return sorted(timesteps)


class ContentAdaptiveNoiseSchedule(nn.Module):
    """
    Generates per-image noise schedules by predicting modulation
    parameters from content descriptors (~0.02M params).

    Each image gets its own beta curve:
        beta_start_i = beta_start_base * exp(s1)
        beta_end_i   = beta_end_base * exp(s2)
    where s1, s2 are predicted from content and clamped to [-0.5, 0.5].

    Complex images get gentler schedules (preserving structure longer).
    Simple images get steeper schedules (can denoise faster).
    """

    def __init__(self, descriptor_dim=64, T=1000,
                 beta_start_base=1e-4, beta_end_base=0.02):
        super().__init__()
        self.T = T
        self.beta_start_base = beta_start_base
        self.beta_end_base = beta_end_base
        self.modulation_range = 0.5  # max log-scale shift

        self.schedule_head = nn.Sequential(
            nn.Linear(descriptor_dim, 32),
            nn.SiLU(),
            nn.Linear(32, 2),
            nn.Tanh(),  # output in [-1, 1], scaled by modulation_range
        )

    def forward(self, content_descriptor):
        """
        Args:
            content_descriptor: [B, descriptor_dim]

        Returns:
            adaptive_betas:         [B, T] per-image beta schedules
            adaptive_alphas_cumprod: [B, T] per-image cumulative alpha products
        """
        modulation = self.schedule_head(content_descriptor)     # [B, 2]
        modulation = modulation * self.modulation_range         # [-0.5, 0.5]

        log_start = modulation[:, 0]  # [B]
        log_end = modulation[:, 1]    # [B]

        beta_start = self.beta_start_base * torch.exp(log_start)  # [B]
        beta_end = self.beta_end_base * torch.exp(log_end)        # [B]

        # Build per-image linear schedules: [B, T]
        t_frac = torch.linspace(
            0, 1, self.T, device=content_descriptor.device
        )  # [T]

        adaptive_betas = (
            beta_start.unsqueeze(1) +
            t_frac.unsqueeze(0) *
            (beta_end.unsqueeze(1) - beta_start.unsqueeze(1))
        )  # [B, T]
        adaptive_betas = adaptive_betas.clamp(1e-5, 0.999)

        adaptive_alphas = 1.0 - adaptive_betas
        adaptive_alphas_cumprod = torch.cumprod(adaptive_alphas, dim=1)

        return adaptive_betas, adaptive_alphas_cumprod
