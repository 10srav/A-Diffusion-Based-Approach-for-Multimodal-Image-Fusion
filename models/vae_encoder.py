"""
VAE Encoder/Decoder for Stable Diffusion
Converts images to/from latent space (512×512 → 64×64×4)
Includes color preservation utilities
"""
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
from typing import Union, Tuple, Optional
import colorsys


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

    def transfer_color(
        self,
        fused_image: Image.Image,
        visible_image: Image.Image,
        strength: float = 0.7
    ) -> Image.Image:
        """
        Transfer color from visible image to fused image using LAB color space.
        Preserves luminance from fused image, takes color from visible.

        Args:
            fused_image: Fused grayscale/desaturated image
            visible_image: Original visible image for color reference
            strength: Color transfer strength (0-1)

        Returns:
            Color-enhanced fused image
        """
        # Ensure same size
        if fused_image.size != visible_image.size:
            visible_image = visible_image.resize(fused_image.size, Image.LANCZOS)

        # Convert to numpy arrays
        fused_np = np.array(fused_image).astype(np.float32) / 255.0
        vis_np = np.array(visible_image).astype(np.float32) / 255.0

        # Convert RGB to YCbCr-like space for color transfer
        # Y = luminance, Cb/Cr = chrominance
        def rgb_to_ycbcr(img):
            y = 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]
            cb = 0.564 * (img[..., 2] - y)
            cr = 0.713 * (img[..., 0] - y)
            return y, cb, cr

        def ycbcr_to_rgb(y, cb, cr):
            r = y + 1.402 * cr
            g = y - 0.344 * cb - 0.714 * cr
            b = y + 1.772 * cb
            return np.stack([r, g, b], axis=-1)

        # Get luminance from fused, color from visible
        fused_y, fused_cb, fused_cr = rgb_to_ycbcr(fused_np)
        vis_y, vis_cb, vis_cr = rgb_to_ycbcr(vis_np)

        # Blend chrominance channels
        blended_cb = strength * vis_cb + (1 - strength) * fused_cb
        blended_cr = strength * vis_cr + (1 - strength) * fused_cr

        # Reconstruct with fused luminance and blended color
        result = ycbcr_to_rgb(fused_y, blended_cb, blended_cr)

        # Clip and convert back
        result = np.clip(result, 0, 1)
        result = (result * 255).astype(np.uint8)

        return Image.fromarray(result)

    def enhance_contrast(
        self,
        image: Image.Image,
        factor: float = 1.2
    ) -> Image.Image:
        """
        Enhance contrast of image.

        Args:
            image: Input image
            factor: Contrast enhancement factor (>1 increases contrast)

        Returns:
            Contrast-enhanced image
        """
        from PIL import ImageEnhance
        enhancer = ImageEnhance.Contrast(image)
        return enhancer.enhance(factor)

    def adjust_brightness(
        self,
        image: Image.Image,
        factor: float = 1.1
    ) -> Image.Image:
        """
        Adjust brightness of image.

        Args:
            image: Input image
            factor: Brightness factor (>1 increases brightness)

        Returns:
            Brightness-adjusted image
        """
        from PIL import ImageEnhance
        enhancer = ImageEnhance.Brightness(image)
        return enhancer.enhance(factor)
