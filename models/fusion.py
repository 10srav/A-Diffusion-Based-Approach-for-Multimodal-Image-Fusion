"""
Fusion Module with Appearance Injection
Implements the fusion rule during denoising (Eq. 13-14 from paper)
"""
import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional
from tqdm import tqdm


class FusionModule:
    """
    Fusion module that generates visible-style fused images.
    Implements appearance injection during the denoising process.
    """
    
    def __init__(
        self,
        unet: nn.Module,
        scheduler,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16
    ):
        """
        Initialize Fusion Module.
        
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
        
        # Fusion parameters from paper
        self.T = 80    # Total diffusion steps
        self.T1 = 70   # IR injection cutoff
        self.T2 = 40   # VIS refinement cutoff
        self.omega = 2.0  # Guidance balance (Eq. 14)
        
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
    def generate_fused_image(
        self,
        feature_memory: Dict[int, Dict[str, torch.Tensor]],
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_uncond: torch.Tensor,
        guidance_scale: float = 7.5,
        verbose: bool = True
    ) -> torch.Tensor:
        """
        Generate fused image using stored features.
        Implements Eq. 12-14 from the paper.
        
        Args:
            feature_memory: Stored K, V, z features from inversion
            encoder_hidden_states: Conditional text embeddings
            encoder_hidden_states_uncond: Unconditional text embeddings
            guidance_scale: CFG scale (default: 7.5, paper uses omega=2)
            verbose: Show progress bar
            
        Returns:
            Fused image latent
        """
        # Set timesteps
        self.scheduler.set_timesteps(self.T, device=self.device)
        timesteps = self.scheduler.timesteps
        
        # Get starting noise from visible latent (Eq. 12: z_fuse = z_vis)
        last_step = max(feature_memory.keys())
        z_vis_T = feature_memory[last_step].get('z_vis')
        
        if z_vis_T is None:
            # Initialize with random noise if no visible features
            latent_shape = (1, 4, 64, 64)
            x_fuse = torch.randn(latent_shape, device=self.device, dtype=self.dtype)
        else:
            # Start with visible noise structure
            mu_vis_T = feature_memory[last_step].get('mu_vis', torch.zeros_like(z_vis_T))
            sigma_T = self._get_sigma(last_step)
            x_fuse = mu_vis_T + sigma_T * z_vis_T
        
        # Progress bar
        iterator = tqdm(timesteps, desc="Fusion Generation") if verbose else timesteps
        
        for i, t in enumerate(iterator):
            t_val = t.item()
            
            # Apply fusion rule (Eq. 13)
            fusion_stage = self._get_fusion_stage(t_val)
            
            # Get z for this step based on fusion stage (Eq. 12)
            z_fuse = self._get_fused_z(t_val, feature_memory, fusion_stage)
            
            # Classifier-free guidance (Eq. 14)
            # f_t(x) = omega * f_t(x, C, Att_fuse) + (1 - omega) * f_t(x, C, empty)
            
            # Conditional forward pass
            noise_pred_cond = self.unet(
                x_fuse,
                t,
                encoder_hidden_states=encoder_hidden_states,
                return_dict=False
            )[0]
            
            # Unconditional forward pass
            noise_pred_uncond = self.unet(
                x_fuse,
                t,
                encoder_hidden_states=encoder_hidden_states_uncond,
                return_dict=False
            )[0]
            
            # Apply CFG (simplified version of Eq. 14)
            noise_pred = noise_pred_uncond + self.omega * (noise_pred_cond - noise_pred_uncond)
            
            # Get alpha values
            alpha_bar_t = self._get_alpha_bar(t_val)
            alpha_bar_prev = self._get_alpha_bar(t_val - 1) if t_val > 0 else 1.0
            
            # Predict x0
            pred_x0 = (x_fuse - (1 - alpha_bar_t) ** 0.5 * noise_pred) / (alpha_bar_t ** 0.5)
            
            # Compute direction
            sigma_t = self._get_sigma(t_val)
            direction = ((1 - alpha_bar_prev - sigma_t ** 2) ** 0.5) * noise_pred
            
            # Compute mu
            mu_fuse = pred_x0 * (alpha_bar_prev ** 0.5) + direction
            
            # Update x_fuse with fused z
            if z_fuse is not None:
                x_fuse = mu_fuse + sigma_t * z_fuse
            else:
                # Use random noise if no stored z
                z_random = torch.randn_like(x_fuse)
                x_fuse = mu_fuse + sigma_t * z_random
                
        return x_fuse
    
    def _get_fusion_stage(self, t: int) -> str:
        """
        Determine fusion stage based on timestep (Eq. 13).
        
        Args:
            t: Current timestep
            
        Returns:
            Stage: 'ir', 'vis', or 'default'
        """
        if t > self.T1:
            # Early stage: Inject IR features
            return 'ir'
        elif t > self.T2:
            # Middle stage: Inject VIS features
            return 'vis'
        else:
            # Late stage: Default denoising
            return 'default'
    
    def _get_fused_z(
        self,
        t: int,
        feature_memory: Dict,
        stage: str
    ) -> Optional[torch.Tensor]:
        """
        Get appropriate z based on fusion stage.
        
        Args:
            t: Current timestep
            feature_memory: Stored features
            stage: Fusion stage ('ir', 'vis', 'default')
            
        Returns:
            z tensor for this step
        """
        if t not in feature_memory:
            return None
            
        mem = feature_memory[t]
        
        if stage == 'ir':
            # Use visible-style infrared z
            return mem.get('z_inf_vis', mem.get('z_vis'))
        elif stage == 'vis':
            # Use visible z
            return mem.get('z_vis')
        else:
            # Default: use visible z for consistency
            return mem.get('z_vis')
    
    def set_fusion_params(self, T: int = 80, T1: int = 70, T2: int = 40, omega: float = 2.0):
        """
        Set fusion parameters.
        
        Args:
            T: Total diffusion steps
            T1: IR injection cutoff step
            T2: VIS refinement cutoff step
            omega: Guidance balance
        """
        self.T = T
        self.T1 = T1
        self.T2 = T2
        self.omega = omega


class VisibleStyleConverter:
    """
    Converts infrared images to visible-style images.
    Used for generating visible-like infrared output.
    """
    
    def __init__(
        self,
        unet: nn.Module,
        scheduler,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16
    ):
        """
        Initialize converter.
        
        Args:
            unet: Stable Diffusion U-Net
            scheduler: Diffusion scheduler
            device: Device to run on
            dtype: Data type for inference
        """
        self.unet = unet
        self.scheduler = scheduler
        self.device = device
        self.dtype = dtype
        
    @torch.no_grad()
    def convert_ir_to_visible_style(
        self,
        feature_memory: Dict[int, Dict[str, torch.Tensor]],
        encoder_hidden_states: torch.Tensor,
        verbose: bool = True
    ) -> torch.Tensor:
        """
        Convert infrared to visible-style image using stored features.
        
        Args:
            feature_memory: Stored features from inversion
            encoder_hidden_states: Text embeddings
            verbose: Show progress bar
            
        Returns:
            Visible-style infrared image latent
        """
        num_steps = max(feature_memory.keys())
        self.scheduler.set_timesteps(num_steps, device=self.device)
        timesteps = self.scheduler.timesteps
        
        # Start from visible-guided infrared noise
        last_step = max(feature_memory.keys())
        z_inf_vis = feature_memory[last_step].get('z_inf_vis')
        mu_inf = feature_memory[last_step].get('mu_inf', torch.zeros_like(z_inf_vis))
        
        # Initialize
        x_t = mu_inf + self._get_sigma(last_step) * z_inf_vis
        
        iterator = tqdm(timesteps, desc="IR→VIS Conversion") if verbose else timesteps
        
        for i, t in enumerate(iterator):
            t_val = t.item()
            
            # Get stored z
            z_t = feature_memory[t_val].get('z_inf_vis', feature_memory[t_val].get('z_vis'))
            
            # Forward pass
            noise_pred = self.unet(
                x_t,
                t,
                encoder_hidden_states=encoder_hidden_states,
                return_dict=False
            )[0]
            
            # DDPM step
            alpha_bar_t = self.scheduler.alphas_cumprod[t_val].item()
            alpha_bar_prev = self.scheduler.alphas_cumprod[t_val - 1].item() if t_val > 0 else 1.0
            
            pred_x0 = (x_t - (1 - alpha_bar_t) ** 0.5 * noise_pred) / (alpha_bar_t ** 0.5)
            
            sigma_t = self._get_sigma(t_val)
            direction = ((1 - alpha_bar_prev - sigma_t ** 2) ** 0.5) * noise_pred
            
            mu = pred_x0 * (alpha_bar_prev ** 0.5) + direction
            
            if z_t is not None:
                x_t = mu + sigma_t * z_t
            else:
                x_t = mu + sigma_t * torch.randn_like(x_t)
                
        return x_t
    
    def _get_sigma(self, timestep: int) -> float:
        """Get sigma at timestep."""
        alpha_bar = self.scheduler.alphas_cumprod[timestep].item()
        alpha_bar_prev = self.scheduler.alphas_cumprod[timestep - 1].item() if timestep > 0 else 1.0
        beta = 1 - alpha_bar / alpha_bar_prev
        sigma = ((1 - alpha_bar_prev) / (1 - alpha_bar)) * beta
        return sigma ** 0.5
