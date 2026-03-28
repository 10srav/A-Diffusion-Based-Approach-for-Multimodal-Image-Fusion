"""
Adaptive Diffusion Fusion: Main Pipeline
Trained diffusion model for IR-VIS image fusion.
Supports single-pair and batch (M3FD dataset) processing via CLI.
"""
import os
import argparse
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Optional, List, Dict
from tqdm import tqdm
import torchvision.transforms.functional as TF

from models.adaptive_fusion_net import (
    AdaptiveFusionModel, LatentAdaptiveFusionModel, create_pseudo_gt,
)
from models.adaptive_diffusion import GaussianDiffusion
from models.content_analyzer import (
    ContentAnalyzer,
    AdaptiveTimestepSelector,
    ContentAdaptiveNoiseSchedule,
)
from models.feature_memory import FeatureMemoryBank, DualModalityFeatureMemory
from models.adaptive_sampler import AdaptiveDDIMSampler
from models.vae import FusionVAE


class AdaptiveDiffusionFusion:
    """
    Adaptive Diffusion Fusion pipeline.

    Auto-detects checkpoint version:
      - v1: Standard DDPM/DDIM sampling
      - v2: Adaptive diffusion with content-aware schedule,
            DDIM inversion, feature memory, adaptive step count
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cuda",
        ddim_steps: int = 50,
        image_size: int = 256,
        min_steps: Optional[int] = None,
        max_steps: Optional[int] = None,
    ):
        self.device = device
        self.ddim_steps = ddim_steps
        self.image_size = image_size
        self.min_steps = min_steps
        self.max_steps = max_steps

        if device == "cuda" and not torch.cuda.is_available():
            print("CUDA not available, falling back to CPU")
            self.device = "cpu"

        print("Loading Adaptive Diffusion Fusion model...")

        self.model = AdaptiveFusionModel().to(self.device)
        self.diffusion = GaussianDiffusion(
            num_timesteps=1000, beta_schedule='linear'
        )

        ckpt = torch.load(
            checkpoint_path, map_location=self.device, weights_only=False
        )

        # Detect version: 1=standard, 2=adaptive pixel, 3=adaptive latent (VAE)
        self.version = ckpt.get('version', 1)

        # Build the right model for this version
        if self.version >= 3:
            # v3: Latent-space model with VAE
            self.model = LatentAdaptiveFusionModel(
                latent_channels=4, latent_size=32
            ).to(self.device)
        else:
            self.model = AdaptiveFusionModel().to(self.device)

        # Load backbone weights
        if 'ema_state_dict' in ckpt:
            self.model.load_state_dict(
                ckpt['ema_state_dict'], strict=False
            )
            print("  Loaded EMA weights")
        elif 'model_state_dict' in ckpt:
            self.model.load_state_dict(
                ckpt['model_state_dict'], strict=False
            )
            print("  Loaded model weights")
        else:
            self.model.load_state_dict(ckpt, strict=False)
            print("  Loaded raw state dict")

        self.model.eval()

        # Initialize adaptive modules for v2+ checkpoints
        self.adaptive_sampler = None
        self.vae = None

        if self.version >= 2:
            print("  Loading adaptive diffusion modules...")
            ca = ContentAnalyzer(descriptor_dim=64).to(self.device)
            ts_kwargs = {'descriptor_dim': 64}
            if self.min_steps is not None:
                ts_kwargs['min_steps'] = self.min_steps
            if self.max_steps is not None:
                ts_kwargs['max_steps'] = self.max_steps
            ts = AdaptiveTimestepSelector(**ts_kwargs).to(self.device)
            ns = ContentAdaptiveNoiseSchedule(descriptor_dim=64).to(self.device)

            ca.load_state_dict(ckpt['content_analyzer_state_dict'])
            ts.load_state_dict(ckpt['timestep_selector_state_dict'])
            ns.load_state_dict(ckpt['noise_schedule_state_dict'])

            ca.eval()
            ts.eval()
            ns.eval()

            # v3: VAE + DualModalityFeatureMemory
            if self.version >= 3:
                self.vae = FusionVAE(latent_channels=4).to(self.device)
                self.vae.load_state_dict(ckpt['vae_state_dict'])
                self.vae.eval()
                print("  VAE loaded (latent-space mode)")

                fm = DualModalityFeatureMemory().to(self.device)
                fm.load_state_dict(ckpt['feature_memory_state_dict'])
                fm.eval()
            else:
                # v2: pixel-space with single FeatureMemoryBank
                fm = FeatureMemoryBank().to(self.device)
                fm.load_state_dict(ckpt['feature_memory_state_dict'])
                fm.eval()

            self.adaptive_sampler = AdaptiveDDIMSampler(
                base_diffusion=self.diffusion,
                content_analyzer=ca,
                timestep_selector=ts,
                noise_schedule=ns,
                feature_memory=fm,
                vae=self.vae,  # None for v2, FusionVAE for v3
            )
            print("  Adaptive modules loaded")

        epoch = ckpt.get('epoch', '?')
        loss = ckpt.get('loss', '?')
        mode_names = {1: 'Standard', 2: 'Adaptive (pixel)', 3: 'Adaptive (latent+VAE)'}
        print(f"  Checkpoint version: v{self.version} ({mode_names.get(self.version, '?')})")
        print(f"  Checkpoint epoch: {epoch}, loss: {loss}")
        print("Adaptive Diffusion Fusion ready!")

    def load_image(self, path: str) -> torch.Tensor:
        """Load image, resize to model size, normalize to [-1, 1]."""
        img = Image.open(path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img = img.resize((self.image_size, self.image_size), Image.LANCZOS)
        tensor = TF.normalize(TF.to_tensor(img), [0.5] * 3, [0.5] * 3)
        return tensor.unsqueeze(0).to(self.device)

    @staticmethod
    def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
        """Convert [-1,1] tensor to PIL Image."""
        img = tensor.squeeze(0).clamp(-1, 1)
        img = (img + 1) / 2  # to [0, 1]
        img = img.cpu()
        return TF.to_pil_image(img)

    @torch.no_grad()
    def fuse(
        self,
        vis_image_path: str,
        ir_image_path: str,
        seed: Optional[int] = None,
        verbose: bool = True,
        strength: float = 0.95,
    ) -> Image.Image:
        """
        Fuse a visible and infrared image pair.

        v1: SDEdit refinement with standard DDIM sampling.
        v2: Adaptive diffusion with content-aware schedule, DDIM inversion
            with feature memory, and adaptive reverse sampling.

        Args:
            vis_image_path: Path to visible image
            ir_image_path: Path to infrared image
            seed: Random seed for reproducibility
            verbose: Show progress
            strength: Noise strength for v1 (ignored in v2, auto-adapted)

        Returns:
            Fused PIL image
        """
        if seed is not None:
            torch.manual_seed(seed)

        if verbose:
            print(f"  VIS: {vis_image_path}")
            print(f"  IR:  {ir_image_path}")
            print(f"  Mode: {'Adaptive (v2)' if self.adaptive_sampler else 'Standard (v1)'}")

        ir = self.load_image(ir_image_path)
        vis = self.load_image(vis_image_path)

        # Create pseudo GT as starting point
        pseudo_gt = create_pseudo_gt(ir, vis)
        shape = (1, 3, self.image_size, self.image_size)

        if self.adaptive_sampler is not None:
            # v2: Adaptive diffusion sampling
            fused_tensor, info = self.adaptive_sampler.sample(
                self.model, shape, ir, vis,
                start_image=pseudo_gt,
                verbose=verbose,
            )
            if verbose:
                print(f"  Adaptive steps: {info['num_steps']}, "
                      f"complexity: {info['complexity_scalar']}")
        else:
            # v1: Standard DDIM with SDEdit
            fused_tensor = self.diffusion.ddim_sample_loop(
                self.model, shape, ir, vis,
                ddim_steps=self.ddim_steps, verbose=verbose,
                start_image=pseudo_gt, start_strength=strength,
            )

        return self.tensor_to_pil(fused_tensor)

    def fuse_batch(
        self,
        data_dir: str,
        output_dir: str,
        seed: int = 42,
        verbose: bool = True,
    ) -> List[Dict]:
        """
        Fuse all image pairs in a dataset directory.

        Args:
            data_dir: Path to directory with vis/ and ir/ subdirs
            output_dir: Output directory for fused images
            seed: Base random seed
            verbose: Show progress

        Returns:
            List of result dictionaries
        """
        from data.m3fd_dataset import M3FDDataset

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        dataset = M3FDDataset(
            root_dir=data_dir,
            image_size=(self.image_size, self.image_size)
        )

        results = []
        iterator = range(len(dataset))
        if verbose:
            iterator = tqdm(iterator, desc="Fusion")

        for idx in iterator:
            pair = dataset[idx]
            name = pair['name']
            out_file = output_path / f"{name}_fused.png"

            if out_file.exists():
                results.append({'name': name, 'output': str(out_file), 'success': True})
                continue

            current_seed = seed + idx

            try:
                fused = self.fuse(
                    vis_image_path=pair['vis_path'],
                    ir_image_path=pair['ir_path'],
                    seed=current_seed,
                    verbose=False
                )
                fused.save(out_file)
                results.append({'name': name, 'output': str(out_file), 'success': True})
            except Exception as e:
                if verbose:
                    print(f"\nError processing {name}: {e}")
                results.append({'name': name, 'error': str(e), 'success': False})

        return results


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Adaptive Diffusion Fusion: IR-VIS Image Fusion"
    )
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained model checkpoint")
    parser.add_argument("--vis_image_path", type=str, default=None,
                        help="Path to visible image (single pair mode)")
    parser.add_argument("--ir_image_path", type=str, default=None,
                        help="Path to infrared image (single pair mode)")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Path to dataset directory (batch mode)")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Output directory")
    parser.add_argument("--ddim_steps", type=int, default=50,
                        help="DDIM sampling steps (default: 50)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device: cuda or cpu")
    parser.add_argument("--min_steps", type=int, default=None,
                        help="Minimum adaptive DDIM steps (default: no limit)")
    parser.add_argument("--max_steps", type=int, default=None,
                        help="Maximum adaptive DDIM steps (default: no limit)")

    args = parser.parse_args()

    if args.data_dir is None and (args.vis_image_path is None or args.ir_image_path is None):
        parser.error("Either --data_dir or both --vis_image_path and --ir_image_path required")

    os.makedirs(args.output_path, exist_ok=True)

    print("=" * 60)
    print("ADAPTIVE DIFFUSION FUSION")
    print("=" * 60)

    fusion = AdaptiveDiffusionFusion(
        checkpoint_path=args.checkpoint,
        device=args.device,
        ddim_steps=args.ddim_steps,
        min_steps=args.min_steps,
        max_steps=args.max_steps,
    )

    if args.data_dir is not None:
        # Batch mode
        results = fusion.fuse_batch(
            data_dir=args.data_dir,
            output_dir=args.output_path,
            seed=args.seed,
        )
        successful = sum(1 for r in results if r['success'])
        print(f"\nComplete: {successful}/{len(results)} images fused")
        print(f"Output: {args.output_path}")
    else:
        # Single pair mode
        fused = fusion.fuse(
            vis_image_path=args.vis_image_path,
            ir_image_path=args.ir_image_path,
            seed=args.seed,
        )
        output_file = os.path.join(args.output_path, "fused.png")
        fused.save(output_file)
        print(f"\nFused image saved: {output_file}")

    print("=" * 60)


if __name__ == "__main__":
    main()
