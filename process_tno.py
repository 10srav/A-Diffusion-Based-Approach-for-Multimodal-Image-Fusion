"""
Batch Processing Script for TNO Dataset
Processes all image pairs from TNO dataset using FusionINV
"""
import os
import argparse
import torch
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

from fusioninv import FusionINV
from data.tno_dataset import TNODataset, download_tno_info


def process_tno_dataset(
    tno_root: str,
    output_dir: str,
    model_path: str = None,
    device: str = "cuda",
    lambda_vis: float = 0.08,
    num_steps: int = 80,
    t1: int = 70,
    t2: int = 40,
    guidance_scale: float = 7.5,
    seed: int = None,
    start_idx: int = 0,
    end_idx: int = None,
    vis_subdir: str = "vis",
    ir_subdir: str = "ir"
):
    """
    Process TNO dataset with FusionINV.

    Args:
        tno_root: Path to TNO dataset root directory
        output_dir: Output directory for fused images
        model_path: Optional local path to SD v1.5 model
        device: Device to use (cuda/cpu)
        lambda_vis: Visible cues strength
        num_steps: Number of diffusion steps
        t1: IR injection cutoff step
        t2: VIS refinement cutoff step
        guidance_scale: CFG guidance scale
        seed: Random seed for reproducibility
        start_idx: Starting index for processing
        end_idx: Ending index (None for all)
        vis_subdir: Subdirectory name for visible images
        ir_subdir: Subdirectory name for infrared images
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Check device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"
        dtype = torch.float32
    else:
        dtype = torch.bfloat16

    # Load TNO dataset
    print(f"\nLoading TNO dataset from: {tno_root}")
    try:
        dataset = TNODataset(
            root_dir=tno_root,
            vis_subdir=vis_subdir,
            ir_subdir=ir_subdir,
            image_size=(512, 512)
        )
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        download_tno_info()
        return

    total_pairs = len(dataset)
    print(f"Found {total_pairs} image pairs")

    # Determine processing range
    if end_idx is None:
        end_idx = total_pairs
    end_idx = min(end_idx, total_pairs)
    start_idx = max(0, start_idx)

    print(f"Processing images {start_idx} to {end_idx - 1}")

    # Initialize FusionINV
    print("\n" + "=" * 60)
    print("Initializing FusionINV...")
    print("=" * 60)

    fusioninv = FusionINV(
        device=device,
        dtype=dtype,
        local_model_path=model_path
    )

    # Set parameters
    fusioninv.set_params(
        lambda_vis=lambda_vis,
        num_steps=num_steps,
        T1=t1,
        T2=t2
    )

    # Process images
    print("\n" + "=" * 60)
    print("Processing TNO Dataset")
    print("=" * 60)

    results = []
    failed = []

    for idx in tqdm(range(start_idx, end_idx), desc="Processing TNO"):
        try:
            # Get image pair
            pair = dataset[idx]
            vis_path = pair['vis_path']
            ir_path = pair['ir_path']
            name = pair['name']

            # Set seed for this image if provided
            current_seed = seed + idx if seed is not None else None

            # Run fusion
            fused_image = fusioninv.fuse(
                vis_image_path=vis_path,
                ir_image_path=ir_path,
                prompt="",
                guidance_scale=guidance_scale,
                seed=current_seed,
                verbose=False
            )

            # Save output
            output_file = output_path / f"{name}_fused.png"
            fused_image.save(output_file)

            results.append({
                'name': name,
                'vis': vis_path,
                'ir': ir_path,
                'output': str(output_file)
            })

        except Exception as e:
            print(f"\nError processing {name}: {e}")
            failed.append({'name': name, 'error': str(e)})

    # Print summary
    print("\n" + "=" * 60)
    print("Processing Complete")
    print("=" * 60)
    print(f"Successful: {len(results)}/{end_idx - start_idx}")
    print(f"Failed: {len(failed)}")
    print(f"Output directory: {output_path}")

    if failed:
        print("\nFailed images:")
        for f in failed:
            print(f"  - {f['name']}: {f['error']}")

    # Save results log
    log_file = output_path / "processing_log.txt"
    with open(log_file, 'w') as f:
        f.write(f"TNO Dataset Processing Log\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n")
        f.write(f"=" * 50 + "\n\n")
        f.write(f"Parameters:\n")
        f.write(f"  lambda_vis: {lambda_vis}\n")
        f.write(f"  num_steps: {num_steps}\n")
        f.write(f"  T1: {t1}\n")
        f.write(f"  T2: {t2}\n")
        f.write(f"  guidance_scale: {guidance_scale}\n")
        f.write(f"  seed: {seed}\n\n")
        f.write(f"Results:\n")
        f.write(f"  Processed: {len(results)}\n")
        f.write(f"  Failed: {len(failed)}\n\n")
        if failed:
            f.write("Failed images:\n")
            for fail in failed:
                f.write(f"  {fail['name']}: {fail['error']}\n")

    print(f"\nLog saved to: {log_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Process TNO dataset with FusionINV"
    )

    # Required arguments
    parser.add_argument(
        "--tno_root",
        type=str,
        required=True,
        help="Path to TNO dataset root directory"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for fused images"
    )

    # Optional arguments
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Local path to SD v1.5 model"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device (cuda/cpu)"
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
        help="Diffusion steps (default: 80)"
    )
    parser.add_argument(
        "--t1",
        type=int,
        default=70,
        help="IR injection cutoff (default: 70)"
    )
    parser.add_argument(
        "--t2",
        type=int,
        default=40,
        help="VIS refinement cutoff (default: 40)"
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=7.5,
        help="CFG scale (default: 7.5)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed"
    )
    parser.add_argument(
        "--start_idx",
        type=int,
        default=0,
        help="Start index for processing"
    )
    parser.add_argument(
        "--end_idx",
        type=int,
        default=None,
        help="End index for processing"
    )
    parser.add_argument(
        "--vis_subdir",
        type=str,
        default="vis",
        help="Visible images subdirectory name"
    )
    parser.add_argument(
        "--ir_subdir",
        type=str,
        default="ir",
        help="Infrared images subdirectory name"
    )

    args = parser.parse_args()

    process_tno_dataset(
        tno_root=args.tno_root,
        output_dir=args.output_dir,
        model_path=args.model_path,
        device=args.device,
        lambda_vis=args.lambda_vis,
        num_steps=args.num_steps,
        t1=args.t1,
        t2=args.t2,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        start_idx=args.start_idx,
        end_idx=args.end_idx,
        vis_subdir=args.vis_subdir,
        ir_subdir=args.ir_subdir
    )


if __name__ == "__main__":
    main()
