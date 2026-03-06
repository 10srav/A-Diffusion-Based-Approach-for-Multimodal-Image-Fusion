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
    device: str = "cuda",
    num_train: int = 200,
    num_test: int = 100,
    epochs: int = 100,
    batch_size: int = 4,
    lr: float = 1e-4,
    ddim_steps: int = 50,
    seed: int = 42,
    skip_setup: bool = False,
    skip_train: bool = False,
    skip_test: bool = False,
):
    """
    Run complete pipeline: Setup -> Training -> Testing.

    Args:
        source_dir: Path to M3FD dataset (containing Ir/ and Vis/)
        device: Device (cuda/cpu)
        num_train: Training images count
        num_test: Test images count
        epochs: Training epochs
        batch_size: Training batch size
        lr: Learning rate
        ddim_steps: DDIM sampling steps for testing
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
    print(f"  Train:       {num_train} images, {epochs} epochs")
    print(f"  Test:        {num_test} images")
    print(f"  Device:      {device}")
    print(f"  Source:      {source_dir}")
    print("=" * 60)

    data_dir = os.path.join(PROJECT_ROOT, "data", "m3fd")
    checkpoint_dir = os.path.join(PROJECT_ROOT, "checkpoints")

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

    # ==================== STEP 2: TRAINING ====================
    if not skip_train:
        print("\n" + "=" * 60)
        print("STEP 2/3: TRAINING PHASE")
        print("=" * 60)

        from run_train import run_training
        run_training(
            data_dir=os.path.join(data_dir, "train"),
            output_dir=os.path.join(PROJECT_ROOT, "output", "training"),
            checkpoint_dir=checkpoint_dir,
            device=device,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            seed=seed,
        )
    else:
        print("\n[Skipping training phase]")

    # ==================== STEP 3: TESTING ====================
    test_metrics, avg_metrics = [], {}

    if not skip_test:
        print("\n" + "=" * 60)
        print("STEP 3/3: TESTING PHASE")
        print("=" * 60)

        from run_test import run_testing
        test_metrics, avg_metrics = run_testing(
            data_dir=os.path.join(data_dir, "test"),
            output_dir=os.path.join(PROJECT_ROOT, "output", "test_results"),
            checkpoint=os.path.join(checkpoint_dir, "best.pt"),
            device=device,
            ddim_steps=ddim_steps,
            seed=seed,
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
    print(f"    ├── training/        (training samples & logs)")
    print(f"    └── test_results/    (100 fused test images + metrics)")
    print(f"        ├── fused/           (fused images)")
    print(f"        ├── test_metrics.csv (per-image metrics)")
    print(f"        └── test_summary.json (average metrics)")
    print(f"  Checkpoints: {checkpoint_dir}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Run complete Adaptive Diffusion Fusion pipeline on M3FD"
    )
    parser.add_argument("--source_dir", type=str, required=True,
                        help="Path to M3FD dataset (containing Ir/ and Vis/)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device: cuda or cpu")
    parser.add_argument("--num_train", type=int, default=200,
                        help="Training images (default: 200)")
    parser.add_argument("--num_test", type=int, default=100,
                        help="Test images (default: 100)")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Training epochs (default: 100)")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Training batch size (default: 4)")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate (default: 1e-4)")
    parser.add_argument("--ddim_steps", type=int, default=50,
                        help="DDIM sampling steps for testing (default: 50)")
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
        device=args.device,
        num_train=args.num_train,
        num_test=args.num_test,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        ddim_steps=args.ddim_steps,
        seed=args.seed,
        skip_setup=args.skip_setup,
        skip_train=args.skip_train,
        skip_test=args.skip_test,
    )


if __name__ == "__main__":
    main()
