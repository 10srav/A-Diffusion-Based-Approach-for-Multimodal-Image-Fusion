"""
Dual-Modality DDIM Inversion for Adaptive Diffusion
Performs SEPARATE inversion for VIS and IR modalities:
  1. VIS inversion first -> stores VIS features in memory
  2. IR inversion guided by VIS features from memory
This ensures IR feature extraction is informed by visible structure.
"""
import torch
from tqdm import tqdm


class DDIMInversion:
    """
    DDIM inversion: deterministically encode x_0 into the noise space x_T.

    Supports both single-image inversion and dual-modality inversion
    where VIS is inverted first and IR inversion uses VIS features as guidance.
    """

    def __init__(self, base_diffusion, feature_memory):
        """
        Args:
            base_diffusion: GaussianDiffusion instance
            feature_memory: DualModalityFeatureMemory instance
        """
        self.base_diffusion = base_diffusion
        self.feature_memory = feature_memory

    @torch.no_grad()
    def _invert_single(self, model, z_0, ir_cond, vis_cond, timesteps,
                        adaptive_alphas_cumprod=None,
                        memory_bank=None, current_modality=None,
                        verbose=False, desc="DDIM Inversion"):
        """
        Core DDIM inversion for a single latent.

        Args:
            model: AdaptiveFusionModel
            z_0: [B, C, H, W] clean latent
            ir_cond, vis_cond: condition images (RGB, for modality encoders)
            timesteps: ascending list of timestep ints
            adaptive_alphas_cumprod: [B, T] per-image schedule or None
            memory_bank: optional memory bank for cross-modal guidance
            current_modality: 'ir' or 'vis' (for feature storage)
            verbose: show progress bar
            desc: progress bar description

        Returns:
            z_T: [B, C, H, W] fully inverted noisy latent
            trajectory: dict {t: z_t}
        """
        B = z_0.shape[0]
        device = z_0.device
        z = z_0.clone()
        trajectory = {timesteps[0]: z_0.clone()}

        pairs = list(zip(timesteps[:-1], timesteps[1:]))
        iterator = tqdm(pairs, desc=desc) if verbose else pairs

        for t_cur, t_next in iterator:
            t_batch = torch.full((B,), t_cur, device=device, dtype=torch.long)

            # Get noise prediction and intermediate features
            noise_pred, intermediates = model.forward_with_intermediates(
                z, t_batch, ir_cond, vis_cond
            )

            # Store features in memory for this modality
            if self.feature_memory is not None and current_modality:
                self.feature_memory.store(
                    t=t_cur,
                    modality=current_modality,
                    features=intermediates,
                )

            # Get alpha values
            alpha_t = self._get_alpha(t_cur, B, device, adaptive_alphas_cumprod)
            alpha_next = self._get_alpha(t_next, B, device, adaptive_alphas_cumprod)

            # Predict z_0 from current state
            pred_z0 = (z - torch.sqrt(1 - alpha_t) * noise_pred) / torch.sqrt(alpha_t)
            pred_z0 = pred_z0.clamp(-1, 1)

            # DDIM forward step (deterministic, eta=0)
            z = (torch.sqrt(alpha_next) * pred_z0 +
                 torch.sqrt(1 - alpha_next) * noise_pred)

            trajectory[t_next] = z.clone()

        return z, trajectory

    @torch.no_grad()
    def invert_dual(self, model, z_vis_0, z_ir_0, ir_cond, vis_cond,
                     timesteps, adaptive_alphas_cumprod=None, verbose=False):
        """
        Dual-modality DDIM inversion matching the client architecture:

        Step 1: VIS Inversion -> Feature Memory (VIS features stored)
        Step 2: IR Inversion guided by VIS (uses stored VIS features)

        Args:
            model: AdaptiveFusionModel
            z_vis_0: [B, C, H, W] visible latent from VAE encoder
            z_ir_0:  [B, C, H, W] infrared latent from VAE encoder
            ir_cond: [B, 3, 256, 256] IR RGB for condition encoders
            vis_cond: [B, 3, 256, 256] VIS RGB for condition encoders
            timesteps: ascending list of timestep ints
            adaptive_alphas_cumprod: [B, T] per-image schedule or None
            verbose: show progress

        Returns:
            z_vis_T: [B, C, H, W] inverted VIS latent
            z_ir_T:  [B, C, H, W] inverted IR latent
            vis_trajectory: dict {t: z_vis_t}
            ir_trajectory:  dict {t: z_ir_t}
        """
        # Clear memory before dual inversion
        if self.feature_memory is not None:
            self.feature_memory.clear()

        # Step 1: VIS Inversion -> stores VIS features
        z_vis_T, vis_trajectory = self._invert_single(
            model, z_vis_0, ir_cond, vis_cond, timesteps,
            adaptive_alphas_cumprod=adaptive_alphas_cumprod,
            current_modality='vis',
            verbose=verbose,
            desc="VIS Inversion",
        )

        # Step 2: IR Inversion guided by VIS features
        # The model can retrieve stored VIS features from memory
        z_ir_T, ir_trajectory = self._invert_single(
            model, z_ir_0, ir_cond, vis_cond, timesteps,
            adaptive_alphas_cumprod=adaptive_alphas_cumprod,
            current_modality='ir',
            verbose=verbose,
            desc="IR Inversion (VIS-guided)",
        )

        return z_vis_T, z_ir_T, vis_trajectory, ir_trajectory

    @torch.no_grad()
    def invert(self, model, x_0, ir_cond, vis_cond, timesteps,
               adaptive_alphas_cumprod=None, verbose=False):
        """
        Single-image DDIM inversion (backward compat with v1).
        """
        if self.feature_memory is not None:
            self.feature_memory.clear()

        return self._invert_single(
            model, x_0, ir_cond, vis_cond, timesteps,
            adaptive_alphas_cumprod=adaptive_alphas_cumprod,
            verbose=verbose,
        )

    def _get_alpha(self, t, B, device, adaptive_alphas_cumprod=None):
        """Get alpha_cumprod at timestep t, either adaptive or base."""
        if adaptive_alphas_cumprod is not None:
            return adaptive_alphas_cumprod[
                torch.arange(B, device=device), t
            ].view(B, 1, 1, 1)
        else:
            alpha = self.base_diffusion.alphas_cumprod[t].to(device)
            return alpha.view(1, 1, 1, 1).expand(B, -1, -1, -1)
