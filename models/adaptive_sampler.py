"""
Adaptive DDIM Sampler for Latent-Space Adaptive Diffusion Fusion

Full pipeline matching the client architecture:
  IR Image -> VAE Enc -> z_ir_0
  VIS Image -> VAE Enc -> z_vis_0
       |
  Adaptive Timestep Module (edge + entropy)
       |
  ADAPTIVE INVERSION:
    VIS Inversion -> Feature Memory
    IR Inversion (guided by VIS)
       |
  Feature Memory (z_t, K, V for both IR & VIS)
       |
  Adaptive Diffusion UNet (noise prediction)
       |
  Adaptive Reverse Diffusion (DDIM + adaptive steps)
       |
  Fused Latent -> VAE Decoder -> Final Fused Image
"""
import torch
import torch.nn.functional as F
from tqdm import tqdm

from .ddim_inversion import DDIMInversion


class AdaptiveDDIMSampler:
    """
    Complete adaptive sampling pipeline with VAE + dual-modality inversion.
    """

    def __init__(self, base_diffusion, content_analyzer, timestep_selector,
                 noise_schedule, feature_memory, vae=None):
        """
        Args:
            base_diffusion: GaussianDiffusion instance
            content_analyzer: ContentAnalyzer module
            timestep_selector: AdaptiveTimestepSelector module
            noise_schedule: ContentAdaptiveNoiseSchedule module
            feature_memory: DualModalityFeatureMemory instance
            vae: FusionVAE instance (for latent-space mode)
        """
        self.base_diffusion = base_diffusion
        self.content_analyzer = content_analyzer
        self.timestep_selector = timestep_selector
        self.noise_schedule = noise_schedule
        self.feature_memory = feature_memory
        self.vae = vae

    @torch.no_grad()
    def sample(self, model, shape, ir_image, vis_image,
               start_image=None, start_strength=0.6,
               verbose=True):
        """
        Full adaptive sampling pipeline.

        If VAE is provided, operates in latent space:
          1. VAE encode IR -> z_ir_0, VAE encode VIS -> z_vis_0
          2. Content analysis on RGB images
          3. Adaptive noise schedule + timestep selection
          4. Dual-modality DDIM inversion (VIS first, then IR guided by VIS)
          5. Adaptive reverse diffusion with memory retrieval
          6. VAE decode -> final fused image

        If no VAE, operates in pixel space (backward compat).

        Args:
            model: AdaptiveFusionModel or LatentAdaptiveFusionModel
            shape: output shape [B, C, H, W] (pixel or latent)
            ir_image: [B, 3, 256, 256] IR RGB image
            vis_image: [B, 3, 256, 256] VIS RGB image
            start_image: optional starting image (pixel space, ignored in VAE mode)
            start_strength: noise strength (used in pixel mode only)
            verbose: show progress

        Returns:
            fused: [B, 3, 256, 256] final fused image (always pixel space)
            info: dict with adaptive sampling metadata
        """
        B = ir_image.shape[0]
        device = ir_image.device
        T = self.base_diffusion.num_timesteps

        # --- Step 1: Content Analysis (always on RGB) ---
        content_desc, complexity_map, complexity_scalar = \
            self.content_analyzer(ir_image, vis_image)

        # --- Step 2: Adaptive Noise Schedule ---
        adaptive_betas, adaptive_alphas_cumprod = \
            self.noise_schedule(content_desc)

        # --- Step 3: Adaptive Timestep Selection ---
        num_steps, density = self.timestep_selector(content_desc)
        n_steps = num_steps[0].item()
        timesteps = self.timestep_selector.build_schedule(
            n_steps, density[0], T=T
        )

        # --- Step 4: Adaptive eta ---
        eta = 0.1 * complexity_scalar.squeeze(-1)

        # --- Step 5: VAE Encode (if latent mode) ---
        if self.vae is not None:
            return self._sample_latent(
                model, ir_image, vis_image, timesteps,
                adaptive_alphas_cumprod, eta, complexity_scalar,
                n_steps, verbose
            )
        else:
            return self._sample_pixel(
                model, shape, ir_image, vis_image, timesteps,
                adaptive_alphas_cumprod, eta, complexity_scalar,
                n_steps, start_image, start_strength, verbose
            )

    @torch.no_grad()
    def _sample_latent(self, model, ir_image, vis_image, timesteps,
                        adaptive_alphas_cumprod, eta, complexity_scalar,
                        n_steps, verbose):
        """
        Latent-space sampling with VAE + dual-modality inversion.
        Matches the client architecture exactly.
        """
        B = ir_image.shape[0]
        device = ir_image.device

        # ======= VAE ENCODE =======
        # IR Image -> VAE Enc -> z_ir_0
        z_ir_0, _, _ = self.vae.encode_ir(ir_image, sample=False)
        # VIS Image -> VAE Enc -> z_vis_0
        z_vis_0, _, _ = self.vae.encode_vis(vis_image, sample=False)

        if verbose:
            print(f"  VAE Encoded: IR {z_ir_0.shape}, VIS {z_vis_0.shape}")

        # ======= ADAPTIVE INVERSION (DUAL MODALITY) =======
        # Adaptive strength based on complexity
        strength = (0.3 + 0.5 * complexity_scalar.squeeze(-1))[0].item()
        start_step = int(len(timesteps) * strength)
        start_step = max(1, min(start_step, len(timesteps) - 1))

        # Inversion timesteps (ascending)
        inversion_max_t = timesteps[start_step]
        inversion_timesteps = [t for t in timesteps if t <= inversion_max_t]
        if 0 not in inversion_timesteps:
            inversion_timesteps = [0] + inversion_timesteps

        # Dual inversion: VIS first, then IR guided by VIS
        inverter = DDIMInversion(self.base_diffusion, self.feature_memory)
        z_vis_T, z_ir_T, vis_traj, ir_traj = inverter.invert_dual(
            model, z_vis_0, z_ir_0, ir_image, vis_image,
            inversion_timesteps, adaptive_alphas_cumprod,
            verbose=verbose,
        )

        if verbose:
            vis_stored = self.feature_memory.num_stored
            print(f"  Feature Memory: VIS={vis_stored['vis']}, IR={vis_stored['ir']} timesteps")

        # ======= FUSE INVERTED LATENTS =======
        # Combine VIS and IR inverted latents as starting point
        # Use complexity-weighted blend (complex scenes weight IR more)
        ir_weight = 0.3 + 0.2 * complexity_scalar.squeeze(-1)  # [0.3, 0.5]
        ir_weight = ir_weight.view(B, 1, 1, 1)
        z_fused_T = (1 - ir_weight) * z_vis_T + ir_weight * z_ir_T

        # ======= ADAPTIVE REVERSE DIFFUSION =======
        active_timesteps = sorted(
            [t for t in timesteps if t <= inversion_max_t],
            reverse=True
        )

        z = z_fused_T
        z = self._reverse_ddim(
            model, z, active_timesteps, ir_image, vis_image,
            adaptive_alphas_cumprod, eta, verbose
        )

        # ======= VAE DECODE =======
        fused_image = self.vae.decode(z)

        info = {
            'num_steps': n_steps,
            'complexity_scalar': complexity_scalar.squeeze(-1).cpu().tolist(),
            'adaptive_strength': strength,
            'eta_mean': eta.mean().item(),
            'latent_shape': list(z_ir_0.shape),
            'mode': 'latent',
        }

        return fused_image, info

    @torch.no_grad()
    def _sample_pixel(self, model, shape, ir_image, vis_image, timesteps,
                       adaptive_alphas_cumprod, eta, complexity_scalar,
                       n_steps, start_image, start_strength, verbose):
        """
        Pixel-space sampling (backward compat, no VAE).
        """
        B = ir_image.shape[0]
        device = ir_image.device

        if start_image is not None:
            strength = (0.3 + 0.5 * complexity_scalar.squeeze(-1))[0].item()
            start_step = int(len(timesteps) * strength)
            start_step = max(1, min(start_step, len(timesteps) - 1))

            inversion_max_t = timesteps[start_step]
            inversion_timesteps = [t for t in timesteps if t <= inversion_max_t]
            if 0 not in inversion_timesteps:
                inversion_timesteps = [0] + inversion_timesteps

            inverter = DDIMInversion(self.base_diffusion, self.feature_memory)
            x_T, trajectory = inverter.invert(
                model, start_image, ir_image, vis_image,
                inversion_timesteps, adaptive_alphas_cumprod,
                verbose=verbose,
            )

            active_timesteps = sorted(
                [t for t in timesteps if t <= inversion_max_t],
                reverse=True
            )
            x = x_T
        else:
            x = torch.randn(shape, device=device)
            active_timesteps = sorted(timesteps, reverse=True)
            self.feature_memory.clear()

        x = self._reverse_ddim(
            model, x, active_timesteps, ir_image, vis_image,
            adaptive_alphas_cumprod, eta, verbose
        )

        info = {
            'num_steps': n_steps,
            'complexity_scalar': complexity_scalar.squeeze(-1).cpu().tolist(),
            'adaptive_strength': strength if start_image is not None else None,
            'eta_mean': eta.mean().item(),
            'mode': 'pixel',
        }

        return x, info

    def _reverse_ddim(self, model, x, active_timesteps, ir_image, vis_image,
                       adaptive_alphas_cumprod, eta, verbose):
        """Core reverse DDIM loop with memory retrieval."""
        B = x.shape[0]
        device = x.device

        timestep_pairs = []
        for i in range(len(active_timesteps) - 1):
            timestep_pairs.append(
                (active_timesteps[i], active_timesteps[i + 1])
            )
        if active_timesteps:
            timestep_pairs.append((active_timesteps[-1], -1))

        iterator = tqdm(timestep_pairs, desc="Adaptive Reverse DDIM") \
            if verbose else timestep_pairs

        for t_cur, t_prev in iterator:
            t_batch = torch.full((B,), t_cur, device=device, dtype=torch.long)

            # Noise prediction with memory-augmented UNet
            noise_pred = model(
                x, t_batch, ir_image, vis_image,
                memory_bank=self.feature_memory,
                current_t=t_cur
            )

            # Per-image adaptive alpha values
            alpha_t = adaptive_alphas_cumprod[
                torch.arange(B, device=device), t_cur
            ].view(B, 1, 1, 1)

            if t_prev >= 0:
                alpha_prev = adaptive_alphas_cumprod[
                    torch.arange(B, device=device), t_prev
                ].view(B, 1, 1, 1)
            else:
                alpha_prev = torch.ones(B, 1, 1, 1, device=device)

            # Predict x_0
            pred_x0 = (x - torch.sqrt(1 - alpha_t) * noise_pred) / \
                       torch.sqrt(alpha_t)
            pred_x0 = pred_x0.clamp(-1, 1)

            # Adaptive DDIM step
            eta_i = eta.view(B, 1, 1, 1)
            sigma = eta_i * torch.sqrt(
                torch.clamp(
                    (1 - alpha_prev) / (1 - alpha_t + 1e-8) *
                    (1 - alpha_t / (alpha_prev + 1e-8)),
                    min=0
                )
            )

            dir_xt = torch.sqrt(
                torch.clamp(1 - alpha_prev - sigma ** 2, min=0)
            ) * noise_pred

            x = torch.sqrt(alpha_prev) * pred_x0 + dir_xt

            if (eta_i > 0).any():
                x = x + sigma * torch.randn_like(x)

        return x
