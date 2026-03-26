"""
Batch Processing Script for TNO Dataset
Processes all image pairs from TNO dataset using Adaptive Diffusion Fusion.
"""
import os
import argparse
import torch
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

from fusioninv import AdaptiveDiffusionFusion
from data.tno_dataset import TNODataset, download_tno_info


def process_tno_dataset(
    tno_root: str,
    output_dir: str,
    checkpoint: str = None,
    device: str = "cuda",
    ddim_steps: int = 50,
    seed: int = None,
    start_idx: int = 0,
    end_idx: int = None,
    vis_subdir: str = "vis",
    ir_subdir: str = "ir"
):
    """
    Process TNO dataset with Adaptive Diffusion Fusion.

    Args:
        tno_root: Path to TNO dataset root directory
        output_dir: Output directory for fused images
        checkpoint: Path to trained model checkpoint
        device: Device to use (cuda/cpu)
        ddim_steps: Number of DDIM sampling steps
        seed: Random seed for reproducibility
        start_idx: Starting index for processing
        end_idx: Ending index (None for all)
        vis_subdir: Subdirectory name for visible images
        ir_subdir: Subdirectory name for infrared images
    """
    # Resolve checkpoint path
    if checkpoint is None:
        project_root = os.path.dirname(os.path.abspath(__file__))
        checkpoint = os.path.join(project_root, "checkpoints", "best.pt")

    if not os.path.exists(checkpoint):
        print(f"Error: Checkpoint not found: {checkpoint}")
        print("Train the model first: python run_train.py")
        return

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Check device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"

    # Load TNO dataset
    print(f"\nLoading TNO dataset from: {tno_root}")
    try:
        dataset = TNODataset(
            root_dir=tno_root,
            vis_subdir=vis_subdir,
            ir_subdir=ir_subdir,
            image_size=(256, 256)
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

    # Initialize Adaptive Diffusion Fusion
    print("\n" + "=" * 60)
    print("Initializing Adaptive Diffusion Fusion...")
    print("=" * 60)

    fusion = AdaptiveDiffusionFusion(
        checkpoint_path=checkpoint,
        device=device,
        ddim_steps=ddim_steps,
    )

    # Process images
    print("\n" + "=" * 60)
    print("Processing TNO Dataset")
    print("=" * 60)

    results = []
    failed = []

    for idx in tqdm(range(start_idx, end_idx), desc="Processing TNO"):
        try:
            pair = dataset[idx]
            vis_path = pair['vis_path']
            ir_path = pair['ir_path']
            name = pair['name']

            # Set seed for this image if provided
            current_seed = seed + idx if seed is not None else None

            # Run fusion
            fused_image = fusion.fuse(
                vis_image_path=vis_path,
                ir_image_path=ir_path,
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
        f.write(f"  checkpoint: {checkpoint}\n")
        f.write(f"  ddim_steps: {ddim_steps}\n")
        f.write(f"  seed: {seed}\n")
        f.write(f"  device: {device}\n\n")
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
        description="Process TNO dataset with Adaptive Diffusion Fusion"
    )

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
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to trained model checkpoint (default: checkpoints/best.pt)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device (cuda/cpu)"
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=50,
        help="DDIM sampling steps (default: 50)"
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
        checkpoint=args.checkpoint,
        device=args.device,
        ddim_steps=args.ddim_steps,
        seed=args.seed,
        start_idx=args.start_idx,
        end_idx=args.end_idx,
        vis_subdir=args.vis_subdir,
        ir_subdir=args.ir_subdir
    )


if __name__ == "__main__":
    main()
