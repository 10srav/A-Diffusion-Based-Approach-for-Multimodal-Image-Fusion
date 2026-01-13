"""
VAE Encoder/Decoder for Stable Diffusion
Converts images to/from latent space (512×512 → 64×64×4)
"""
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
from typing import Union, Tuple


class VAEEncoder:
    """
    VAE Encoder/Decoder wrapper for Stable Diffusion.
    Handles image to latent and latent to image conversion.
    """
    
    def __init__(self, vae, device: str = "cuda", dtype: torch.dtype = torch.bfloat16):
        """
        Initialize VAE encoder with a pre-trained VAE model.
        
        Args:
            vae: Pre-trained VAE from diffusers
            device: Device to run on (cuda/cpu)
            dtype: Data type for inference (bfloat16 for speed)
        """
        self.vae = vae
        self.device = device
        self.dtype = dtype
        self.scaling_factor = 0.18215  # SD v1.5 VAE scaling factor
        
    def preprocess_image(self, image: Union[Image.Image, np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Preprocess image for VAE encoding.
        
        Args:
            image: Input image (PIL, numpy, or tensor)
            
        Returns:
            Preprocessed tensor ready for VAE encoding
        """
        if isinstance(image, Image.Image):
            # Convert PIL to numpy
            image = np.array(image)
            
        if isinstance(image, np.ndarray):
            # Handle grayscale images (IR images)
            if len(image.shape) == 2:
                image = np.stack([image] * 3, axis=-1)
            elif image.shape[-1] == 1:
                image = np.concatenate([image] * 3, axis=-1)
                
            # Normalize to [-1, 1]
            image = image.astype(np.float32) / 255.0
            image = (image - 0.5) * 2.0
            
            # Convert to tensor [B, C, H, W]
            image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
            
        return image.to(device=self.device, dtype=self.dtype)
    
    @torch.no_grad()
    def encode(self, image: Union[Image.Image, np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Encode image to latent space.
        
        Args:
            image: Input image (512×512)
            
        Returns:
            Latent representation (64×64×4)
        """
        # Preprocess
        x = self.preprocess_image(image)
        
        # Encode through VAE
        latent_dist = self.vae.encode(x)
        
        # Get latent sample
        if hasattr(latent_dist, 'latent_dist'):
            latent = latent_dist.latent_dist.sample()
        else:
            latent = latent_dist.sample()
            
        # Apply scaling factor
        latent = latent * self.scaling_factor
        
        return latent
    
    @torch.no_grad()
    def decode(self, latent: torch.Tensor) -> Image.Image:
        """
        Decode latent to image.
        
        Args:
            latent: Latent representation (64×64×4)
            
        Returns:
            Decoded PIL image (512×512)
        """
        # Remove scaling factor
        latent = latent / self.scaling_factor
        
        # Decode through VAE
        decoded = self.vae.decode(latent).sample
        
        # Convert to PIL image
        image = self.postprocess_image(decoded)
        
        return image
    
    def postprocess_image(self, tensor: torch.Tensor) -> Image.Image:
        """
        Convert tensor to PIL image.
        
        Args:
            tensor: Image tensor [B, C, H, W] in [-1, 1]
            
        Returns:
            PIL Image
        """
        # Clamp and normalize to [0, 1]
        tensor = (tensor / 2 + 0.5).clamp(0, 1)
        
        # Convert to numpy
        image = tensor.cpu().float().permute(0, 2, 3, 1).numpy()[0]
        
        # Convert to uint8
        image = (image * 255).round().astype(np.uint8)
        
        return Image.fromarray(image)
    
    def resize_image(self, image: Image.Image, size: Tuple[int, int] = (512, 512)) -> Image.Image:
        """
        Resize image to target size.
        
        Args:
            image: Input PIL image
            size: Target size (width, height)
            
        Returns:
            Resized PIL image
        """
        return image.resize(size, Image.LANCZOS)
