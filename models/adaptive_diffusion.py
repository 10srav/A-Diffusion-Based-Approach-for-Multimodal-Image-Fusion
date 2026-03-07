"""
Gaussian Diffusion Process for Adaptive Image Fusion
Implements DDPM forward/reverse process and DDIM fast sampling.
"""
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm


def linear_beta_schedule(num_timesteps, beta_start=1e-4, beta_end=0.02):
    """Linear noise schedule (Ho et al. 2020)."""
    return torch.linspace(beta_start, beta_end, num_timesteps, dtype=torch.float64)


def cosine_beta_schedule(num_timesteps, s=0.008):
    """Cosine noise schedule (Nichol & Dhariwal 2021)."""
    steps = num_timesteps + 1
    x = torch.linspace(0, num_timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / num_timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


class GaussianDiffusion:
    """
    Gaussian Diffusion process with DDPM training and DDIM sampling.

    Args:
        num_timesteps: Total diffusion timesteps (default: 1000)
        beta_schedule: 'linear' or 'cosine'
        beta_start: Starting beta for linear schedule
        beta_end: Ending beta for linear schedule
    """

    def __init__(
        self,
        num_timesteps=1000,
        beta_schedule='linear',
        beta_start=1e-4,
        beta_end=0.02
    ):
        self.num_timesteps = num_timesteps

        if beta_schedule == 'linear':
            betas = linear_beta_schedule(num_timesteps, beta_start, beta_end)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(num_timesteps)
        else:
            raise ValueError(f"Unknown schedule: {beta_schedule}")

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        # Store as float32 buffers
        self.betas = betas.float()
        self.alphas = alphas.float()
        self.alphas_cumprod = alphas_cumprod.float()
        self.alphas_cumprod_prev = alphas_cumprod_prev.float()

        # Precomputed values for forward process
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod).float()
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod).float()

        # SNR (signal-to-noise ratio) for min-SNR loss weighting
        self.snr = (alphas_cumprod / (1.0 - alphas_cumprod)).float()

        # Precomputed values for reverse process (posterior)
        self.posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        ).float()
        self.posterior_log_variance = torch.log(
            torch.clamp(self.posterior_variance, min=1e-20)
        ).float()
        self.posterior_mean_coef1 = (
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        ).float()
        self.posterior_mean_coef2 = (
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod)
        ).float()

    def _extract(self, arr, t, shape):
        """Extract values from array at timestep indices and reshape."""
        batch_size = t.shape[0]
        out = arr.to(t.device).gather(0, t.long())
        return out.reshape(batch_size, *((1,) * (len(shape) - 1)))

    def q_sample(self, x_0, t, noise=None):
        """
        Forward process: add noise to clean image.

        Args:
            x_0: [B, C, H, W] clean image
            t: [B] timestep indices
            noise: optional pre-sampled noise

        Returns:
            x_t: [B, C, H, W] noisy image at timestep t
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        sqrt_alpha = self._extract(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_alpha = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape)

        return sqrt_alpha * x_0 + sqrt_one_minus_alpha * noise

    def predict_x0_from_noise(self, x_t, t, noise):
        """Reconstruct x_0 from x_t and predicted noise."""
        sqrt_alpha = self._extract(self.sqrt_alphas_cumprod, t, x_t.shape)
        sqrt_one_minus_alpha = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
        return (x_t - sqrt_one_minus_alpha * noise) / sqrt_alpha

    def p_losses(self, model, x_0, t, ir_image, vis_image, noise=None,
                 min_snr_gamma=5.0):
        """
        Compute DDPM training loss with min-SNR weighting.

        Min-SNR weighting (Hang et al. 2023) prevents the loss from being
        dominated by very high noise timesteps where the model can't learn
        useful denoising. It clips the SNR weight at gamma, focusing training
        on mid-range timesteps that matter most for image quality.

        Args:
            model: AdaptiveFusionModel
            x_0: [B, C, H, W] clean fused image (pseudo GT)
            t: [B] timestep indices
            ir_image: [B, 3, H, W] infrared image
            vis_image: [B, 3, H, W] visible image
            noise: optional pre-sampled noise
            min_snr_gamma: SNR clipping value (default: 5.0, set 0 to disable)

        Returns:
            loss: weighted MSE loss between predicted and actual noise
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        x_noisy = self.q_sample(x_0, t, noise=noise)
        noise_pred = model(x_noisy, t, ir_image, vis_image)

        if min_snr_gamma > 0:
            # Per-sample MSE (not reduced)
            mse = F.mse_loss(noise_pred, noise, reduction='none')
            mse = mse.mean(dim=[1, 2, 3])  # [B]

            # min-SNR weight: min(SNR(t), gamma) / SNR(t)
            snr_t = self._extract(self.snr, t, torch.Size([t.shape[0]])).squeeze()
            weight = torch.clamp(snr_t, max=min_snr_gamma) / (snr_t + 1e-8)

            return (weight * mse).mean()
        else:
            return F.mse_loss(noise_pred, noise)

    @torch.no_grad()
    def p_sample(self, model, x_t, t, ir_image, vis_image):
        """
        Single DDPM reverse step: x_t -> x_{t-1}.

        Args:
            model: AdaptiveFusionModel
            x_t: [B, C, H, W] noisy image at step t
            t: [B] current timestep
            ir_image, vis_image: condition images

        Returns:
            x_{t-1}: [B, C, H, W]
        """
        noise_pred = model(x_t, t, ir_image, vis_image)

        # Compute posterior mean
        coef1 = self._extract(self.posterior_mean_coef1, t, x_t.shape)
        coef2 = self._extract(self.posterior_mean_coef2, t, x_t.shape)

        pred_x0 = self.predict_x0_from_noise(x_t, t, noise_pred)
        pred_x0 = torch.clamp(pred_x0, -1, 1)

        mean = coef1 * pred_x0 + coef2 * x_t

        # Add noise for all steps except t=0
        variance = self._extract(self.posterior_variance, t, x_t.shape)
        noise = torch.randn_like(x_t)
        nonzero_mask = (t != 0).float().reshape(-1, *((1,) * (len(x_t.shape) - 1)))

        return mean + nonzero_mask * torch.sqrt(variance) * noise

    @torch.no_grad()
    def p_sample_loop(self, model, shape, ir_image, vis_image, verbose=True):
        """
        Full DDPM reverse process (1000 steps).

        Args:
            model: AdaptiveFusionModel
            shape: Output shape [B, C, H, W]
            ir_image, vis_image: condition images
            verbose: show progress bar

        Returns:
            x_0: [B, C, H, W] generated image
        """
        device = ir_image.device
        x = torch.randn(shape, device=device)

        timesteps = list(reversed(range(self.num_timesteps)))
        iterator = tqdm(timesteps, desc="DDPM Sampling") if verbose else timesteps

        for t_val in iterator:
            t = torch.full((shape[0],), t_val, device=device, dtype=torch.long)
            x = self.p_sample(model, x, t, ir_image, vis_image)

        return x

    def ddim_timesteps(self, ddim_steps=50):
        """Compute DDIM timestep schedule (evenly spaced, includes T-1)."""
        # Use linspace to ensure we include both 0 and T-1 (999)
        # This prevents the mismatch where sampling starts from pure noise (t=T)
        # but the schedule only covers up to t=980
        return np.linspace(0, self.num_timesteps - 1, ddim_steps, dtype=int).tolist()

    @torch.no_grad()
    def ddim_step(self, model, x_t, t_cur, t_prev, ir_image, vis_image, eta=0.0):
        """
        Single DDIM reverse step.

        Args:
            model: AdaptiveFusionModel
            x_t: current noisy image
            t_cur: current timestep
            t_prev: previous timestep (closer to 0)
            ir_image, vis_image: conditions
            eta: DDIM stochasticity (0 = deterministic)

        Returns:
            x_{t_prev}
        """
        B = x_t.shape[0]
        device = x_t.device

        t_batch = torch.full((B,), t_cur, device=device, dtype=torch.long)
        noise_pred = model(x_t, t_batch, ir_image, vis_image)

        # Get alpha values
        alpha_t = self._extract(self.alphas_cumprod, t_batch, x_t.shape)
        if t_prev >= 0:
            t_prev_batch = torch.full((B,), t_prev, device=device, dtype=torch.long)
            alpha_prev = self._extract(self.alphas_cumprod, t_prev_batch, x_t.shape)
        else:
            alpha_prev = torch.ones_like(alpha_t)

        # Predict x_0
        pred_x0 = (x_t - torch.sqrt(1 - alpha_t) * noise_pred) / torch.sqrt(alpha_t)
        pred_x0 = torch.clamp(pred_x0, -1, 1)

        # Compute sigma for stochasticity
        sigma = eta * torch.sqrt((1 - alpha_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_prev))

        # Direction pointing to x_t
        dir_xt = torch.sqrt(1 - alpha_prev - sigma ** 2) * noise_pred

        # DDIM step
        x_prev = torch.sqrt(alpha_prev) * pred_x0 + dir_xt

        if eta > 0:
            x_prev = x_prev + sigma * torch.randn_like(x_t)

        return x_prev

    @torch.no_grad()
    def ddim_sample_loop(self, model, shape, ir_image, vis_image,
                         ddim_steps=50, eta=0.0, verbose=True,
                         start_image=None, start_strength=0.6):
        """
        DDIM fast sampling with optional SDEdit (image-to-image refinement).

        When start_image is provided, uses SDEdit approach: adds noise to
        the start_image up to a mid-range timestep, then denoises from there.
        This produces much cleaner outputs with small models since the model
        refines an existing image rather than generating from pure noise.

        Args:
            model: AdaptiveFusionModel
            shape: Output shape [B, C, H, W]
            ir_image, vis_image: condition images
            ddim_steps: number of DDIM steps
            eta: stochasticity (0 = deterministic)
            verbose: show progress bar
            start_image: [B, C, H, W] optional starting image (e.g. pseudo GT)
            start_strength: fraction of timesteps to denoise (0.6 = start at 60% noise)

        Returns:
            x_0: [B, C, H, W] generated image
        """
        device = ir_image.device
        timesteps = self.ddim_timesteps(ddim_steps)

        if start_image is not None:
            # SDEdit: start from partially noised image
            start_step = int(len(timesteps) * start_strength)
            start_step = max(1, min(start_step, len(timesteps) - 1))
            t_start = timesteps[-(start_step + 1)]

            # Add noise to start_image at t_start
            t_batch = torch.full((shape[0],), t_start, device=device, dtype=torch.long)
            noise = torch.randn(shape, device=device)
            x = self.q_sample(start_image, t_batch, noise=noise)

            # Only use timesteps from t_start downward
            active_timesteps = [t for t in timesteps if t <= t_start]
        else:
            # Standard: start from pure noise
            x = torch.randn(shape, device=device)
            active_timesteps = timesteps

        timestep_pairs = list(zip(reversed(active_timesteps[1:]), reversed(active_timesteps[:-1])))
        timestep_pairs.append((active_timesteps[0], -1))

        iterator = tqdm(timestep_pairs, desc="DDIM Sampling") if verbose else timestep_pairs

        for t_cur, t_prev in iterator:
            x = self.ddim_step(model, x, t_cur, t_prev, ir_image, vis_image, eta=eta)

        return x
