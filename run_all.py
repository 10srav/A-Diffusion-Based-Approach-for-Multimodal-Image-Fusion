"""
Complete Pipeline: Setup + Training + Testing
Runs the entire Adaptive Diffusion Fusion pipeline on M3FD dataset.
Single command: python run_all.py --source_dir /path/to/M3FD
"""
import os
import sys
import argparse
import time
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)


def run_all(
    source_dir: str,
    model_path: str = None,
    device: str = "cuda",
    num_train: int = 200,
    num_test: int = 100,
    lambda_vis: float = 0.08,
    num_steps: int = 80,
    t1: int = 70,
    t2: int = 40,
    guidance_scale: float = 7.5,
    seed: int = 42,
    skip_setup: bool = False,
    skip_train: bool = False,
    skip_test: bool = False
):
    """
    Run complete pipeline: Setup -> Training -> Testing.

    Args:
        source_dir: Path to M3FD dataset (containing Ir/ and Vis/)
        model_path: Optional local path to SD v1.5 model
        device: Device (cuda/cpu)
        num_train: Training images count
        num_test: Test images count
        lambda_vis: Visible cues strength
        num_steps: Diffusion steps
        t1: IR injection cutoff
        t2: VIS refinement cutoff
        guidance_scale: CFG scale
        seed: Random seed
        skip_setup: Skip dataset setup
        skip_train: Skip training phase
        skip_test: Skip testing phase
    """
    start_time = time.time()

    print("=" * 60)
    print("ADAPTIVE DIFFUSION FUSION - COMPLETE PIPELINE")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    print(f"  Dataset:     M3FD")
    print(f"  Train:       {num_train} images")
    print(f"  Test:        {num_test} images")
    print(f"  Device:      {device}")
    print(f"  Source:      {source_dir}")
    print("=" * 60)

    data_dir = os.path.join(PROJECT_ROOT, "data", "m3fd")

    # ==================== STEP 1: SETUP ====================
    if not skip_setup:
        print("\n" + "=" * 60)
        print("STEP 1/3: DATASET SETUP")
        print("=" * 60)

        from setup_m3fd import setup_m3fd
        setup_m3fd(
            source_dir=source_dir,
            download=False,
            num_train=num_train,
            num_test=num_test,
            seed=seed
        )
    else:
        print("\n[Skipping setup - using existing split]")

    train_results, train_failed = [], []
    test_metrics, avg_metrics = [], {}

    # ==================== STEP 2: TRAINING ====================
    if not skip_train:
        print("\n" + "=" * 60)
        print("STEP 2/3: TRAINING PHASE")
        print("=" * 60)

        from run_train import run_training
        train_results, train_failed = run_training(
            data_dir=os.path.join(data_dir, "train"),
            output_dir=os.path.join(PROJECT_ROOT, "output", "train_results"),
            model_path=model_path,
            device=device,
            lambda_vis=lambda_vis,
            num_steps=num_steps,
            t1=t1,
            t2=t2,
            guidance_scale=guidance_scale,
            seed=seed
        )
    else:
        print("\n[Skipping training phase]")

    # ==================== STEP 3: TESTING ====================
    if not skip_test:
        print("\n" + "=" * 60)
        print("STEP 3/3: TESTING PHASE")
        print("=" * 60)

        from run_test import run_testing
        test_metrics, avg_metrics = run_testing(
            data_dir=os.path.join(data_dir, "test"),
            output_dir=os.path.join(PROJECT_ROOT, "output", "test_results"),
            model_path=model_path,
            device=device,
            lambda_vis=lambda_vis,
            num_steps=num_steps,
            t1=t1,
            t2=t2,
            guidance_scale=guidance_scale,
            seed=seed
        )
    else:
        print("\n[Skipping testing phase]")

    # ==================== SUMMARY ====================
    elapsed = time.time() - start_time
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    seconds = int(elapsed % 60)

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE!")
    print("=" * 60)
    print(f"  Total time: {hours}h {minutes}m {seconds}s")
    print(f"  Output directory: {os.path.join(PROJECT_ROOT, 'output')}")
    print(f"    ├── train_results/   (200 fused training images)")
    print(f"    └── test_results/    (100 fused test images + metrics)")
    print(f"        ├── fused/           (fused images)")
    print(f"        ├── test_metrics.csv (per-image metrics)")
    print(f"        └── test_summary.json (average metrics)")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Run complete Adaptive Diffusion Fusion pipeline on M3FD"
    )
    parser.add_argument("--source_dir", type=str, required=True,
                        help="Path to M3FD dataset (containing Ir/ and Vis/)")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Local path to SD v1.5 model")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device: cuda or cpu")
    parser.add_argument("--num_train", type=int, default=200,
                        help="Training images (default: 200)")
    parser.add_argument("--num_test", type=int, default=100,
                        help="Test images (default: 100)")
    parser.add_argument("--lambda_vis", type=float, default=0.08,
                        help="Visible cues strength")
    parser.add_argument("--num_steps", type=int, default=80,
                        help="Diffusion steps")
    parser.add_argument("--t1", type=int, default=70,
                        help="IR injection cutoff")
    parser.add_argument("--t2", type=int, default=40,
                        help="VIS refinement cutoff")
    parser.add_argument("--guidance_scale", type=float, default=7.5,
                        help="CFG scale")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--skip_setup", action="store_true",
                        help="Skip dataset setup")
    parser.add_argument("--skip_train", action="store_true",
                        help="Skip training phase")
    parser.add_argument("--skip_test", action="store_true",
                        help="Skip testing phase")

    args = parser.parse_args()

    run_all(
        source_dir=args.source_dir,
        model_path=args.model_path,
        device=args.device,
        num_train=args.num_train,
        num_test=args.num_test,
        lambda_vis=args.lambda_vis,
        num_steps=args.num_steps,
        t1=args.t1,
        t2=args.t2,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        skip_setup=args.skip_setup,
        skip_train=args.skip_train,
        skip_test=args.skip_test
    )


if __name__ == "__main__":
    main()
