"""
DDPM Inversion with Visible Cues Guidance
Implements visible-guided inversion for infrared images (Eq. 5-11 from paper)
"""
import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional, List
from tqdm import tqdm


class DDPMInversion:
    """
    DDPM Inversion module with visible cues guidance.
    Inverts both visible and infrared images into noise latent space.
    """
    
    def __init__(
        self,
        unet: nn.Module,
        scheduler,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16
    ):
        """
        Initialize DDPM Inversion.
        
        Args:
            unet: Stable Diffusion U-Net
            scheduler: Diffusion scheduler (DDPMScheduler)
            device: Device to run on
            dtype: Data type for inference
        """
        self.unet = unet
        self.scheduler = scheduler
        self.device = device
        self.dtype = dtype
        
        # Feature memory storage
        self.feature_memory: Dict[int, Dict[str, torch.Tensor]] = {}
        
        # Diffusion parameters
        self.num_inference_steps = 80  # T = 80
        
    def _get_alpha_bar(self, timestep: int) -> float:
        """Get cumulative alpha at timestep."""
        return self.scheduler.alphas_cumprod[timestep].item()
    
    def _get_sigma(self, timestep: int) -> float:
        """Get sigma at timestep."""
        alpha_bar = self._get_alpha_bar(timestep)
        alpha_bar_prev = self._get_alpha_bar(timestep - 1) if timestep > 0 else 1.0
        beta = 1 - alpha_bar / alpha_bar_prev
        sigma = ((1 - alpha_bar_prev) / (1 - alpha_bar)) * beta
        return sigma ** 0.5
    
    @torch.no_grad()
    def invert_visible(
        self,
        x_vis_0: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        verbose: bool = True
    ) -> Dict[int, Dict[str, torch.Tensor]]:
        """
        Invert visible image to noise latent space.
        Implements Eq. 5-7 from the paper.
        
        Args:
            x_vis_0: Visible image latent (64×64×4)
            encoder_hidden_states: Text embeddings
            verbose: Show progress bar
            
        Returns:
            Feature memory with stored K, V, z at each timestep
        """
        # Set timesteps
        self.scheduler.set_timesteps(self.num_inference_steps, device=self.device)
        timesteps = self.scheduler.timesteps
        
        # Initialize with encoded image
        x_vis_t = x_vis_0.clone()
        
        # Clear previous memory
        self.feature_memory.clear()
        
        # Progress bar
        iterator = tqdm(timesteps, desc="Visible Inversion") if verbose else timesteps
        
        for i, t in enumerate(iterator):
            # Get alpha values
            alpha_bar_t = self._get_alpha_bar(t.item())
            alpha_bar_prev = self._get_alpha_bar(t.item() - 1) if t.item() > 0 else 1.0
            
            # Predict noise (Eq. 5)
            noise_pred = self.unet(
                x_vis_t,
                t,
                encoder_hidden_states=encoder_hidden_states,
                return_dict=False
            )[0]
            
            # Predict x0 from noise
            # P(f_t(x_t)) = (x_t - sqrt(1-alpha_bar_t) * f_t(x_t)) / sqrt(alpha_bar_t)
            pred_x0 = (x_vis_t - (1 - alpha_bar_t) ** 0.5 * noise_pred) / (alpha_bar_t ** 0.5)
            
            # Direction pointing to x_t
            # D(f_t(x_t)) = sqrt(1 - alpha_bar_{t-1} - sigma_t^2) * f_t(x_t)
            sigma_t = self._get_sigma(t.item())
            direction = ((1 - alpha_bar_prev - sigma_t ** 2) ** 0.5) * noise_pred
            
            # Compute mu (Eq. 5)
            mu_vis_t = pred_x0 * (alpha_bar_prev ** 0.5) + direction
            
            # Compute z_vis (Eq. 6)
            # z_vis_t = (sqrt(alpha_bar_{t-1}) * x_vis_0 + sqrt(1-alpha_bar_{t-1}) * eps - mu_vis_t) / sigma_t
            noise = torch.randn_like(x_vis_t)
            target = (alpha_bar_prev ** 0.5) * x_vis_0 + ((1 - alpha_bar_prev) ** 0.5) * noise
            z_vis_t = (target - mu_vis_t) / (sigma_t + 1e-8)
            
            # Store in feature memory
            self.feature_memory[t.item()] = {
                'z_vis': z_vis_t.detach().clone(),
                'mu_vis': mu_vis_t.detach().clone(),
                'pred_x0_vis': pred_x0.detach().clone(),
            }
            
            # Update x_t for next step (Eq. 7)
            x_vis_t = mu_vis_t + sigma_t * z_vis_t
            
        return self.feature_memory
    
    @torch.no_grad()
    def invert_infrared_with_visible_cues(
        self,
        x_inf_0: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        lambda_vis: float = 0.08,
        verbose: bool = True
    ) -> Dict[int, Dict[str, torch.Tensor]]:
        """
        Invert infrared image with visible cues guidance.
        Implements Eq. 8-11 from the paper.
        
        Args:
            x_inf_0: Infrared image latent (64×64×4)
            encoder_hidden_states: Text embeddings
            lambda_vis: Visible cues strength (default: 0.08)
            verbose: Show progress bar
            
        Returns:
            Updated feature memory with IR features
        """
        if not self.feature_memory:
            raise ValueError("Must run invert_visible first to get visible cues")
            
        # Set timesteps
        self.scheduler.set_timesteps(self.num_inference_steps, device=self.device)
        timesteps = self.scheduler.timesteps
        
        # Initialize with encoded infrared image
        x_inf_t = x_inf_0.clone()
        
        # Progress bar
        iterator = tqdm(timesteps, desc="IR Inversion (Visible Guided)") if verbose else timesteps
        
        for i, t in enumerate(iterator):
            t_val = t.item()
            
            # Get alpha values
            alpha_bar_t = self._get_alpha_bar(t_val)
            alpha_bar_prev = self._get_alpha_bar(t_val - 1) if t_val > 0 else 1.0
            
            # Predict noise for infrared
            noise_pred = self.unet(
                x_inf_t,
                t,
                encoder_hidden_states=encoder_hidden_states,
                return_dict=False
            )[0]
            
            # Predict x0 from noise
            pred_x0 = (x_inf_t - (1 - alpha_bar_t) ** 0.5 * noise_pred) / (alpha_bar_t ** 0.5)
            
            # Direction pointing to x_t
            sigma_t = self._get_sigma(t_val)
            direction = ((1 - alpha_bar_prev - sigma_t ** 2) ** 0.5) * noise_pred
            
            # Compute mu_inf (same as visible)
            mu_inf_t = pred_x0 * (alpha_bar_prev ** 0.5) + direction
            
            # Compute z_inf
            noise = torch.randn_like(x_inf_t)
            target = (alpha_bar_prev ** 0.5) * x_inf_0 + ((1 - alpha_bar_prev) ** 0.5) * noise
            z_inf_t = (target - mu_inf_t) / (sigma_t + 1e-8)
            
            # Get visible cues (Eq. 8-11)
            z_vis_t = self.feature_memory[t_val]['z_vis']
            
            # Apply visible cues guidance (Eq. 11)
            # z_inf->vis = z_inf + lambda * z_vis
            z_inf_vis_t = z_inf_t + lambda_vis * z_vis_t
            
            # Store infrared features
            self.feature_memory[t_val].update({
                'z_inf': z_inf_t.detach().clone(),
                'z_inf_vis': z_inf_vis_t.detach().clone(),
                'mu_inf': mu_inf_t.detach().clone(),
                'pred_x0_inf': pred_x0.detach().clone(),
            })
            
            # Update x_t with visible-guided noise
            x_inf_t = mu_inf_t + sigma_t * z_inf_vis_t
            
        return self.feature_memory
    
    def get_inverted_latent(self, source: str = 'vis') -> Optional[torch.Tensor]:
        """
        Get the final inverted latent (at T).
        
        Args:
            source: 'vis' for visible, 'inf' for infrared
            
        Returns:
            Inverted latent at final timestep
        """
        if not self.feature_memory:
            return None
            
        # Get the last timestep
        last_step = max(self.feature_memory.keys())
        
        if source == 'vis':
            return self.feature_memory[last_step].get('z_vis')
        else:
            return self.feature_memory[last_step].get('z_inf_vis')
    
    def get_all_z(self, source: str = 'vis') -> Dict[int, torch.Tensor]:
        """
        Get all z values at each timestep.
        
        Args:
            source: 'vis', 'inf', or 'inf_vis'
            
        Returns:
            Dictionary mapping timestep to z tensor
        """
        key = f'z_{source}'
        return {t: mem[key] for t, mem in self.feature_memory.items() if key in mem}
