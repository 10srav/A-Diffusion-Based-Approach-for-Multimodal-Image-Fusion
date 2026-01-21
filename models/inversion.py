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
        Invert visible image to noise latent space using DDIM inversion.
        Goes from x_0 -> x_T by reversing the denoising process.

        Args:
            x_vis_0: Visible image latent (64×64×4)
            encoder_hidden_states: Text embeddings
            verbose: Show progress bar

        Returns:
            Feature memory with stored features at each timestep
        """
        # Set timesteps for denoising (T -> 0)
        self.scheduler.set_timesteps(self.num_inference_steps, device=self.device)
        # Reverse for inversion (0 -> T)
        timesteps = reversed(self.scheduler.timesteps)
        timesteps_list = list(timesteps)

        # Initialize with encoded image
        x_vis_t = x_vis_0.clone()

        # Clear previous memory
        self.feature_memory.clear()

        # Progress bar
        iterator = tqdm(timesteps_list, desc="Visible Inversion") if verbose else timesteps_list

        for i, t in enumerate(iterator):
            t_val = t.item()

            # Get alpha values
            alpha_bar_t = self._get_alpha_bar(t_val)

            # Predict noise using UNet
            noise_pred = self.unet(
                x_vis_t,
                t,
                encoder_hidden_states=encoder_hidden_states,
                return_dict=False
            )[0]

            # Predict x0 from current x_t and noise
            pred_x0 = (x_vis_t - (1 - alpha_bar_t) ** 0.5 * noise_pred) / (alpha_bar_t ** 0.5 + 1e-8)

            # Get next timestep (going forward in diffusion)
            if i < len(timesteps_list) - 1:
                next_t = timesteps_list[i + 1].item()
                alpha_bar_next = self._get_alpha_bar(next_t)
            else:
                # Last step - use highest noise level
                alpha_bar_next = self.scheduler.alphas_cumprod[-1].item()
                next_t = len(self.scheduler.alphas_cumprod) - 1

            # DDIM inversion: x_{t+1} = sqrt(alpha_bar_{t+1}) * pred_x0 + sqrt(1 - alpha_bar_{t+1}) * noise_pred
            x_vis_next = (alpha_bar_next ** 0.5) * pred_x0 + ((1 - alpha_bar_next) ** 0.5) * noise_pred

            # Store features for this timestep
            self.feature_memory[t_val] = {
                'z_vis': noise_pred.detach().clone(),
                'x_t_vis': x_vis_t.detach().clone(),
                'pred_x0_vis': pred_x0.detach().clone(),
            }

            # Update for next iteration
            x_vis_t = x_vis_next

        # Store final noisy latent
        self.final_latent_vis = x_vis_t.detach().clone()

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
        Invert infrared image with visible cues guidance using DDIM inversion.

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

        # Set timesteps for denoising (T -> 0)
        self.scheduler.set_timesteps(self.num_inference_steps, device=self.device)
        # Reverse for inversion (0 -> T)
        timesteps = reversed(self.scheduler.timesteps)
        timesteps_list = list(timesteps)

        # Initialize with encoded infrared image
        x_inf_t = x_inf_0.clone()

        # Progress bar
        iterator = tqdm(timesteps_list, desc="IR Inversion (Visible Guided)") if verbose else timesteps_list

        for i, t in enumerate(iterator):
            t_val = t.item()

            # Get alpha values
            alpha_bar_t = self._get_alpha_bar(t_val)

            # Predict noise for infrared
            noise_pred_inf = self.unet(
                x_inf_t,
                t,
                encoder_hidden_states=encoder_hidden_states,
                return_dict=False
            )[0]

            # Predict x0 from noise
            pred_x0 = (x_inf_t - (1 - alpha_bar_t) ** 0.5 * noise_pred_inf) / (alpha_bar_t ** 0.5 + 1e-8)

            # Get visible noise prediction for guidance
            z_vis_t = self.feature_memory[t_val]['z_vis']

            # Apply visible cues guidance to noise prediction
            # Blend infrared noise with visible noise
            noise_pred_guided = noise_pred_inf + lambda_vis * (z_vis_t - noise_pred_inf)

            # Get next timestep (going forward in diffusion)
            if i < len(timesteps_list) - 1:
                next_t = timesteps_list[i + 1].item()
                alpha_bar_next = self._get_alpha_bar(next_t)
            else:
                alpha_bar_next = self.scheduler.alphas_cumprod[-1].item()
                next_t = len(self.scheduler.alphas_cumprod) - 1

            # DDIM inversion with guided noise
            x_inf_next = (alpha_bar_next ** 0.5) * pred_x0 + ((1 - alpha_bar_next) ** 0.5) * noise_pred_guided

            # Store infrared features
            self.feature_memory[t_val].update({
                'z_inf': noise_pred_inf.detach().clone(),
                'z_inf_vis': noise_pred_guided.detach().clone(),
                'x_t_inf': x_inf_t.detach().clone(),
                'pred_x0_inf': pred_x0.detach().clone(),
            })

            # Update for next iteration
            x_inf_t = x_inf_next

        # Store final noisy latent
        self.final_latent_inf = x_inf_t.detach().clone()

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
