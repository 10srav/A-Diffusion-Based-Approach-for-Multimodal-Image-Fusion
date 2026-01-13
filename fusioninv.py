"""
FusionINV: Main Pipeline
Training-free multimodal image fusion using Stable Diffusion v1.5
"""
import os
import argparse
import torch
from PIL import Image
from typing import Optional, Tuple
from tqdm import tqdm

# Diffusers imports
from diffusers import StableDiffusionPipeline, DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer

# Local imports
from models.vae_encoder import VAEEncoder
from models.inversion import DDPMInversion
from models.fusion import FusionModule


class FusionINV:
    """
    FusionINV: A Diffusion-Based Approach for Multimodal Image Fusion
    
    Training-free method that produces visible-style fused images
    using Stable Diffusion v1.5 inversion.
    """
    
    def __init__(
        self,
        model_id: str = "stable-diffusion-v1-5/stable-diffusion-v1-5",
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        local_model_path: Optional[str] = None
    ):
        """
        Initialize FusionINV pipeline.
        
        Args:
            model_id: HuggingFace model ID or local path
            device: Device to run on (cuda/cpu)
            dtype: Data type (bfloat16 for speed)
            local_model_path: Optional local path to pre-downloaded model
        """
        self.device = device
        self.dtype = dtype
        
        print("Loading Stable Diffusion v1.5...")
        
        # Use local path if provided
        model_path = local_model_path if local_model_path else model_id
        
        # Load pipeline
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_path,
            torch_dtype=dtype,
            safety_checker=None,
            requires_safety_checker=False
        ).to(device)
        
        # Extract components
        self.vae = self.pipe.vae
        self.unet = self.pipe.unet
        self.tokenizer = self.pipe.tokenizer
        self.text_encoder = self.pipe.text_encoder
        
        # Create custom scheduler for inversion
        self.scheduler = DDPMScheduler.from_pretrained(
            model_path,
            subfolder="scheduler"
        )
        
        # Initialize modules
        self.vae_encoder = VAEEncoder(self.vae, device, dtype)
        self.inversion = DDPMInversion(self.unet, self.scheduler, device, dtype)
        self.fusion = FusionModule(self.unet, self.scheduler, device, dtype)
        
        # Default parameters
        self.lambda_vis = 0.08  # Visible cues strength
        self.num_steps = 80    # T = 80
        self.T1 = 70           # IR injection cutoff
        self.T2 = 40           # VIS refinement cutoff
        self.omega = 2.0       # Guidance balance
        
        print("FusionINV initialized successfully!")
        
    def set_params(
        self,
        lambda_vis: float = 0.08,
        num_steps: int = 80,
        T1: int = 70,
        T2: int = 40,
        omega: float = 2.0
    ):
        """Set fusion parameters."""
        self.lambda_vis = lambda_vis
        self.num_steps = num_steps
        self.T1 = T1
        self.T2 = T2
        self.omega = omega
        
        # Update fusion module
        self.fusion.set_fusion_params(num_steps, T1, T2, omega)
        self.inversion.num_inference_steps = num_steps
        
    def _get_text_embedding(self, prompt: str) -> torch.Tensor:
        """Get text embeddings from CLIP text encoder."""
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        )
        
        with torch.no_grad():
            text_embeddings = self.text_encoder(
                text_inputs.input_ids.to(self.device)
            )[0]
            
        return text_embeddings.to(dtype=self.dtype)
    
    def _load_and_preprocess(self, image_path: str, size: Tuple[int, int] = (512, 512)) -> Image.Image:
        """Load and preprocess image."""
        image = Image.open(image_path)
        
        # Convert grayscale to RGB if needed
        if image.mode == 'L':
            image = image.convert('RGB')
        elif image.mode != 'RGB':
            image = image.convert('RGB')
            
        # Resize
        image = image.resize(size, Image.LANCZOS)
        
        return image
    
    @torch.no_grad()
    def fuse(
        self,
        vis_image_path: str,
        ir_image_path: str,
        prompt: str = "",
        guidance_scale: float = 7.5,
        seed: Optional[int] = None,
        verbose: bool = True
    ) -> Image.Image:
        """
        Fuse visible and infrared images.
        
        Args:
            vis_image_path: Path to visible image
            ir_image_path: Path to infrared image
            prompt: Optional text prompt for guidance
            guidance_scale: CFG scale
            seed: Random seed for reproducibility
            verbose: Show progress
            
        Returns:
            Fused PIL image
        """
        # Set seed
        if seed is not None:
            torch.manual_seed(seed)
            
        if verbose:
            print(f"Processing images...")
            print(f"  Visible: {vis_image_path}")
            print(f"  Infrared: {ir_image_path}")
            print(f"  Lambda: {self.lambda_vis}, Steps: {self.num_steps}")
            
        # Load images
        vis_image = self._load_and_preprocess(vis_image_path)
        ir_image = self._load_and_preprocess(ir_image_path)
        
        # Encode to latent space
        if verbose:
            print("Encoding images to latent space...")
        x_vis_0 = self.vae_encoder.encode(vis_image)
        x_inf_0 = self.vae_encoder.encode(ir_image)
        
        # Get text embeddings
        if verbose:
            print("Getting text embeddings...")
        text_emb = self._get_text_embedding(prompt if prompt else "")
        uncond_emb = self._get_text_embedding("")
        
        # Step 1: Invert visible image
        if verbose:
            print("\nStep 1: Inverting visible image...")
        feature_memory = self.inversion.invert_visible(
            x_vis_0, text_emb, verbose=verbose
        )
        
        # Step 2: Invert infrared with visible cues guidance
        if verbose:
            print("\nStep 2: Inverting infrared with visible cues...")
        feature_memory = self.inversion.invert_infrared_with_visible_cues(
            x_inf_0, text_emb, lambda_vis=self.lambda_vis, verbose=verbose
        )
        
        # Step 3: Generate fused image
        if verbose:
            print("\nStep 3: Generating fused image...")
        fused_latent = self.fusion.generate_fused_image(
            feature_memory,
            text_emb,
            uncond_emb,
            guidance_scale=guidance_scale,
            verbose=verbose
        )
        
        # Decode to image
        if verbose:
            print("\nDecoding fused latent to image...")
        fused_image = self.vae_encoder.decode(fused_latent)
        
        return fused_image
    
    @torch.no_grad()
    def convert_ir_to_visible(
        self,
        ir_image_path: str,
        vis_image_path: str,
        prompt: str = "",
        seed: Optional[int] = None,
        verbose: bool = True
    ) -> Image.Image:
        """
        Convert infrared image to visible-style using visible cues.
        
        Args:
            ir_image_path: Path to infrared image
            vis_image_path: Path to visible image (for style guidance)
            prompt: Optional text prompt
            seed: Random seed
            verbose: Show progress
            
        Returns:
            Visible-style infrared image
        """
        from models.fusion import VisibleStyleConverter
        
        # Set seed
        if seed is not None:
            torch.manual_seed(seed)
            
        # Load images
        vis_image = self._load_and_preprocess(vis_image_path)
        ir_image = self._load_and_preprocess(ir_image_path)
        
        # Encode
        x_vis_0 = self.vae_encoder.encode(vis_image)
        x_inf_0 = self.vae_encoder.encode(ir_image)
        
        # Get embeddings
        text_emb = self._get_text_embedding(prompt if prompt else "")
        
        # Inversion
        feature_memory = self.inversion.invert_visible(x_vis_0, text_emb, verbose=verbose)
        feature_memory = self.inversion.invert_infrared_with_visible_cues(
            x_inf_0, text_emb, lambda_vis=self.lambda_vis, verbose=verbose
        )
        
        # Convert
        converter = VisibleStyleConverter(self.unet, self.scheduler, self.device, self.dtype)
        converted_latent = converter.convert_ir_to_visible_style(
            feature_memory, text_emb, verbose=verbose
        )
        
        # Decode
        converted_image = self.vae_encoder.decode(converted_latent)
        
        return converted_image


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="FusionINV: Diffusion-Based Multimodal Image Fusion"
    )
    
    # Required arguments
    parser.add_argument(
        "--vis_image_path",
        type=str,
        required=True,
        help="Path to visible image"
    )
    parser.add_argument(
        "--ir_image_path",
        type=str,
        required=True,
        help="Path to infrared image"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Output directory"
    )
    
    # Optional arguments
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Local path to SD v1.5 model (optional)"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="",
        help="Text prompt for guided fusion"
    )
    parser.add_argument(
        "--lambda_vis",
        type=float,
        default=0.08,
        help="Visible cues strength (default: 0.08)"
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=80,
        help="Number of diffusion steps (default: 80)"
    )
    parser.add_argument(
        "--t1",
        type=int,
        default=70,
        help="IR injection cutoff step (default: 70)"
    )
    parser.add_argument(
        "--t2",
        type=int,
        default=40,
        help="VIS refinement cutoff step (default: 40)"
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=7.5,
        help="CFG guidance scale (default: 7.5)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (cuda/cpu)"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_path, exist_ok=True)
    
    # Check CUDA availability
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = "cpu"
        dtype = torch.float32
    else:
        dtype = torch.bfloat16
        
    # Initialize FusionINV
    print("=" * 60)
    print("FusionINV: Diffusion-Based Multimodal Image Fusion")
    print("=" * 60)
    
    fusioninv = FusionINV(
        device=args.device,
        dtype=dtype,
        local_model_path=args.model_path
    )
    
    # Set parameters
    fusioninv.set_params(
        lambda_vis=args.lambda_vis,
        num_steps=args.num_steps,
        T1=args.t1,
        T2=args.t2
    )
    
    # Run fusion
    print("\n" + "=" * 60)
    print("Starting fusion process...")
    print("=" * 60)
    
    fused_image = fusioninv.fuse(
        vis_image_path=args.vis_image_path,
        ir_image_path=args.ir_image_path,
        prompt=args.prompt,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        verbose=True
    )
    
    # Save output
    output_file = os.path.join(args.output_path, "fused.png")
    fused_image.save(output_file)
    
    print("\n" + "=" * 60)
    print(f"Fusion complete! Output saved to: {output_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()
