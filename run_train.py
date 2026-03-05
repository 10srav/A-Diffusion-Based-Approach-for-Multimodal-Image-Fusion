"""
Training Script for M3FD Dataset
Processes 200 training images using the Adaptive Diffusion Fusion model.
Run from command line: python run_train.py
"""
import os
import sys
import argparse
import torch
import json
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from fusioninv import FusionINV
from data.m3fd_dataset import M3FDDataset


def run_training(
    data_dir: str = None,
    output_dir: str = None,
    model_path: str = None,
    device: str = "cuda",
    lambda_vis: float = 0.08,
    num_steps: int = 80,
    t1: int = 70,
    t2: int = 40,
    guidance_scale: float = 7.5,
    seed: int = 42,
    start_idx: int = 0,
    end_idx: int = None
):
    """
    Process M3FD training set (200 images) with adaptive diffusion fusion.

    Args:
        data_dir: Path to M3FD train split (containing vis/ and ir/)
        output_dir: Output directory for fused images
        model_path: Optional local path to SD v1.5 model
        device: Device (cuda/cpu)
        lambda_vis: Visible cues strength
        num_steps: Diffusion steps
        t1: IR injection cutoff
        t2: VIS refinement cutoff
        guidance_scale: CFG scale
        seed: Random seed
        start_idx: Start index (for resuming)
        end_idx: End index (None = all)
    """
    # Default paths
    if data_dir is None:
        data_dir = os.path.join(PROJECT_ROOT, "data", "m3fd", "train")
    if output_dir is None:
        output_dir = os.path.join(PROJECT_ROOT, "output", "train_results")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Check CUDA
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"
        dtype = torch.float32
    else:
        dtype = torch.bfloat16 if device == "cuda" else torch.float32

    # Load dataset
    print("=" * 60)
    print("ADAPTIVE DIFFUSION FUSION - TRAINING PHASE")
    print("Dataset: M3FD (200 images)")
    print("=" * 60)

    try:
        dataset = M3FDDataset(
            root_dir=data_dir,
            image_size=(512, 512)
        )
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nPlease run setup first:")
        print("  python setup_m3fd.py --source_dir /path/to/M3FD")
        sys.exit(1)

    total = len(dataset)
    if end_idx is None:
        end_idx = total
    end_idx = min(end_idx, total)
    start_idx = max(0, start_idx)

    print(f"Processing images {start_idx} to {end_idx - 1} ({end_idx - start_idx} images)")
    print(f"Device: {device}")
    print(f"Parameters: lambda={lambda_vis}, steps={num_steps}, T1={t1}, T2={t2}")

    # Initialize model
    print("\nLoading Stable Diffusion v1.5...")
    fusioninv = FusionINV(
        device=device,
        dtype=dtype,
        local_model_path=model_path
    )
    fusioninv.set_params(
        lambda_vis=lambda_vis,
        num_steps=num_steps,
        T1=t1,
        T2=t2
    )

    # Process images
    print("\n" + "=" * 60)
    print("Processing training images...")
    print("=" * 60)

    results = []
    failed = []

    for idx in tqdm(range(start_idx, end_idx), desc="Training Fusion"):
        pair = dataset[idx]
        name = pair['name']
        current_seed = seed + idx if seed is not None else None

        try:
            fused_image = fusioninv.fuse(
                vis_image_path=pair['vis_path'],
                ir_image_path=pair['ir_path'],
                prompt="",
                guidance_scale=guidance_scale,
                seed=current_seed,
                verbose=False
            )

            out_file = output_path / f"{name}_fused.png"
            fused_image.save(out_file)

            results.append({
                'name': name,
                'vis_path': pair['vis_path'],
                'ir_path': pair['ir_path'],
                'output': str(out_file),
                'success': True
            })

        except Exception as e:
            print(f"\nError processing {name}: {e}")
            failed.append({'name': name, 'error': str(e)})

    # Save results log
    log = {
        'phase': 'training',
        'timestamp': datetime.now().isoformat(),
        'dataset': 'M3FD',
        'total_processed': len(results),
        'total_failed': len(failed),
        'parameters': {
            'lambda_vis': lambda_vis,
            'num_steps': num_steps,
            'T1': t1,
            'T2': t2,
            'guidance_scale': guidance_scale,
            'seed': seed,
            'device': device
        },
        'results': results,
        'failed': failed
    }

    log_file = output_path / "train_results.json"
    with open(log_file, 'w') as f:
        json.dump(log, f, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print("TRAINING PHASE COMPLETE")
    print("=" * 60)
    print(f"  Successful: {len(results)}/{end_idx - start_idx}")
    print(f"  Failed:     {len(failed)}")
    print(f"  Output:     {output_dir}")
    print(f"  Log:        {log_file}")

    if failed:
        print(f"\n  Failed images:")
        for f_item in failed:
            print(f"    - {f_item['name']}: {f_item['error']}")

    print("\nNext step: python run_test.py")
    print("=" * 60)

    return results, failed


def main():
    parser = argparse.ArgumentParser(
        description="Process M3FD training set with Adaptive Diffusion Fusion"
    )
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Path to M3FD train data (default: data/m3fd/train)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: output/train_results)")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Local path to SD v1.5 model")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device: cuda or cpu (default: cuda)")
    parser.add_argument("--lambda_vis", type=float, default=0.08,
                        help="Visible cues strength (default: 0.08)")
    parser.add_argument("--num_steps", type=int, default=80,
                        help="Diffusion steps (default: 80)")
    parser.add_argument("--t1", type=int, default=70,
                        help="IR injection cutoff (default: 70)")
    parser.add_argument("--t2", type=int, default=40,
                        help="VIS refinement cutoff (default: 40)")
    parser.add_argument("--guidance_scale", type=float, default=7.5,
                        help="CFG scale (default: 7.5)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--start_idx", type=int, default=0,
                        help="Start index for resuming (default: 0)")
    parser.add_argument("--end_idx", type=int, default=None,
                        help="End index (default: all)")

    args = parser.parse_args()

    run_training(
        data_dir=args.data_dir,
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
        end_idx=args.end_idx
    )


if __name__ == "__main__":
    main()
