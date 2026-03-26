"""
Testing Script for M3FD Dataset
Processes 100 test images and computes quality metrics.
Run from command line: python run_test.py --checkpoint checkpoints/best.pt
"""
import os
import sys
import argparse
import torch
import json
import csv
import numpy as np
import cv2
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from fusioninv import AdaptiveDiffusionFusion
from data.m3fd_dataset import M3FDDataset


# ===================== QUALITY METRICS =====================

def mutual_information(a, b):
    """Compute Mutual Information between two grayscale images."""
    h, _, _ = np.histogram2d(a.ravel(), b.ravel(), bins=256, range=[[0, 255], [0, 255]])
    pxy = h / np.sum(h)
    px = np.sum(pxy, axis=1)
    py = np.sum(pxy, axis=0)
    px_py = px[:, None] * py[None, :]
    nz = pxy > 0
    return float(np.sum(pxy[nz] * np.log2(pxy[nz] / px_py[nz])))


def correlation_coefficient(a, b):
    """Compute Correlation Coefficient between two images."""
    return float(np.corrcoef(a.flatten(), b.flatten())[0, 1])


def entropy_value(img):
    """Compute Shannon Entropy of an image."""
    from scipy.stats import entropy
    hist, _ = np.histogram(img.flatten(), bins=256, density=True)
    hist = hist[hist > 0]
    return float(entropy(hist))


def spatial_frequency(img):
    """Compute Spatial Frequency of an image."""
    rf = np.diff(img.astype(np.float64), axis=0)
    cf = np.diff(img.astype(np.float64), axis=1)
    return float(np.sqrt(np.mean(rf ** 2) + np.mean(cf ** 2)))


def average_gradient(img):
    """Compute Average Gradient of an image."""
    gx, gy = np.gradient(img.astype(np.float64))
    return float(np.mean(np.sqrt(gx ** 2 + gy ** 2)))


def standard_deviation(img):
    """Compute Standard Deviation of an image."""
    return float(np.std(img.astype(np.float64)))


def ssim_metric(a, b):
    """Compute SSIM between two images."""
    from skimage.metrics import structural_similarity
    return float(structural_similarity(a, b, data_range=255))


def vif_metric(a, b):
    """Compute VIF (Visual Information Fidelity)."""
    try:
        from sewar.full_ref import vifp
        return float(vifp(a, b))
    except ImportError:
        return _simple_vif(a, b)


def _simple_vif(ref, dist):
    """Simple VIF approximation when sewar is not available."""
    sigma1_sq = cv2.GaussianBlur(ref.astype(np.float64) ** 2, (11, 11), 1.5) - \
                cv2.GaussianBlur(ref.astype(np.float64), (11, 11), 1.5) ** 2
    sigma2_sq = cv2.GaussianBlur(dist.astype(np.float64) ** 2, (11, 11), 1.5) - \
                cv2.GaussianBlur(dist.astype(np.float64), (11, 11), 1.5) ** 2
    sigma12 = cv2.GaussianBlur(ref.astype(np.float64) * dist.astype(np.float64), (11, 11), 1.5) - \
              cv2.GaussianBlur(ref.astype(np.float64), (11, 11), 1.5) * \
              cv2.GaussianBlur(dist.astype(np.float64), (11, 11), 1.5)
    eps = 1e-10
    sigma_nsq = 2.0
    g = sigma12 / (sigma1_sq + eps)
    sv_sq = sigma2_sq - g * sigma12
    g = np.maximum(g, 0)
    sv_sq = np.maximum(sv_sq, eps)
    num = np.log2(1 + g ** 2 * sigma1_sq / (sv_sq + eps) + eps)
    den = np.log2(1 + sigma1_sq / sigma_nsq)
    vif = np.sum(num + eps) / (np.sum(den) + eps)
    return float(np.clip(vif, 0, 1))


def compute_metrics(ir_img, vis_img, fused_img):
    """
    Compute all quality metrics for a fused image.

    Args:
        ir_img: Infrared image (grayscale numpy array)
        vis_img: Visible image (grayscale numpy array)
        fused_img: Fused image (grayscale numpy array)

    Returns:
        Dictionary of metric names to values
    """
    h, w = fused_img.shape[:2]
    if ir_img.shape[:2] != (h, w):
        ir_img = cv2.resize(ir_img, (w, h))
    if vis_img.shape[:2] != (h, w):
        vis_img = cv2.resize(vis_img, (w, h))

    metrics = {
        'MI_IR_F': mutual_information(ir_img, fused_img),
        'MI_VIS_F': mutual_information(vis_img, fused_img),
        'SSIM_IR_F': ssim_metric(ir_img, fused_img),
        'SSIM_VIS_F': ssim_metric(vis_img, fused_img),
        'CC_IR_F': correlation_coefficient(ir_img, fused_img),
        'CC_VIS_F': correlation_coefficient(vis_img, fused_img),
        'EN': entropy_value(fused_img),
        'SD': standard_deviation(fused_img),
        'SF': spatial_frequency(fused_img),
        'AG': average_gradient(fused_img),
        'VIF_IR_F': vif_metric(ir_img, fused_img),
        'VIF_VIS_F': vif_metric(vis_img, fused_img),
    }

    return metrics


# ===================== MAIN TEST PIPELINE =====================

def run_testing(
    data_dir=None,
    output_dir=None,
    checkpoint=None,
    device="cuda",
    ddim_steps=50,
    seed=42,
    start_idx=0,
    end_idx=None,
    skip_fusion=False,
):
    """
    Process M3FD test set (100 images) and compute quality metrics.

    Args:
        data_dir: Path to M3FD test split
        output_dir: Output directory
        checkpoint: Path to trained model checkpoint
        device: Device (cuda/cpu)
        ddim_steps: DDIM sampling steps
        seed: Random seed
        start_idx: Start index
        end_idx: End index
        skip_fusion: Skip fusion, only compute metrics on existing outputs
    """
    if data_dir is None:
        data_dir = os.path.join(PROJECT_ROOT, "data", "m3fd", "test")
    if output_dir is None:
        output_dir = os.path.join(PROJECT_ROOT, "output", "test_results")
    if checkpoint is None:
        checkpoint = os.path.join(PROJECT_ROOT, "checkpoints", "best.pt")

    output_path = Path(output_dir)
    fused_dir = output_path / "fused"
    fused_dir.mkdir(parents=True, exist_ok=True)

    # Check CUDA
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"

    # Load dataset
    print("=" * 60)
    print("ADAPTIVE DIFFUSION FUSION - TESTING PHASE")
    print("Dataset: M3FD (100 images)")
    print("=" * 60)

    try:
        dataset = M3FDDataset(
            root_dir=data_dir,
            image_size=(256, 256)
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

    # ---- STEP 1: Run Fusion (unless skipping) ----
    if not skip_fusion:
        print("\n--- Step 1: Generating fused images ---")

        if not os.path.exists(checkpoint):
            print(f"\nError: Checkpoint not found: {checkpoint}")
            print("Train the model first: python run_train.py")
            sys.exit(1)

        fusion = AdaptiveDiffusionFusion(
            checkpoint_path=checkpoint,
            device=device,
            ddim_steps=ddim_steps,
        )

        for idx in tqdm(range(start_idx, end_idx), desc="Test Fusion"):
            pair = dataset[idx]
            name = pair['name']
            out_file = fused_dir / f"{name}_fused.png"

            if out_file.exists():
                continue

            current_seed = seed + idx

            try:
                fused_image = fusion.fuse(
                    vis_image_path=pair['vis_path'],
                    ir_image_path=pair['ir_path'],
                    seed=current_seed,
                    verbose=False
                )
                fused_image.save(out_file)
            except Exception as e:
                print(f"\nError fusing {name}: {e}")

        del fusion
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        print("\n--- Skipping fusion (using existing outputs) ---")

    # ---- STEP 2: Compute Quality Metrics ----
    print("\n--- Step 2: Computing quality metrics ---")

    all_metrics = []
    metric_names = ['MI_IR_F', 'MI_VIS_F', 'SSIM_IR_F', 'SSIM_VIS_F',
                    'CC_IR_F', 'CC_VIS_F', 'EN', 'SD', 'SF', 'AG',
                    'VIF_IR_F', 'VIF_VIS_F']

    for idx in tqdm(range(start_idx, end_idx), desc="Computing Metrics"):
        pair = dataset[idx]
        name = pair['name']
        fused_file = fused_dir / f"{name}_fused.png"

        if not fused_file.exists():
            print(f"\n  Warning: Fused image not found for {name}, skipping metrics")
            continue

        ir_img = cv2.imread(pair['ir_path'], cv2.IMREAD_GRAYSCALE)
        vis_img = cv2.imread(pair['vis_path'], cv2.IMREAD_GRAYSCALE)
        fused_img = cv2.imread(str(fused_file), cv2.IMREAD_GRAYSCALE)

        if ir_img is None or vis_img is None or fused_img is None:
            print(f"\n  Warning: Could not load images for {name}")
            continue

        try:
            metrics = compute_metrics(ir_img, vis_img, fused_img)
            metrics['name'] = name
            all_metrics.append(metrics)
        except Exception as e:
            print(f"\n  Error computing metrics for {name}: {e}")

    # ---- STEP 3: Aggregate and Save Results ----
    print("\n--- Step 3: Aggregating results ---")

    if len(all_metrics) == 0:
        print("No metrics computed. Check if fused images exist.")
        return [], {}

    avg_metrics = {}
    for metric in metric_names:
        values = [m[metric] for m in all_metrics if metric in m]
        if values:
            avg_metrics[metric] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values))
            }

    csv_file = output_path / "test_metrics.csv"
    with open(csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['name'] + metric_names)
        writer.writeheader()
        for m in all_metrics:
            row = {k: f"{m.get(k, 0):.6f}" if k != 'name' else m[k] for k in ['name'] + metric_names}
            writer.writerow(row)

    summary = {
        'phase': 'testing',
        'timestamp': datetime.now().isoformat(),
        'dataset': 'M3FD',
        'num_images': len(all_metrics),
        'parameters': {
            'checkpoint': checkpoint,
            'ddim_steps': ddim_steps,
            'seed': seed,
            'device': device,
        },
        'average_metrics': avg_metrics
    }

    json_file = output_path / "test_summary.json"
    with open(json_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print("TEST RESULTS - QUALITY METRICS")
    print("=" * 60)
    print(f"{'Metric':<18} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
    print("-" * 60)

    for metric in metric_names:
        if metric in avg_metrics:
            m = avg_metrics[metric]
            print(f"{metric:<18} {m['mean']:>10.4f} {m['std']:>10.4f} {m['min']:>10.4f} {m['max']:>10.4f}")

    print("-" * 60)
    print(f"Total images evaluated: {len(all_metrics)}")
    print(f"\nResults saved to:")
    print(f"  Per-image CSV: {csv_file}")
    print(f"  Summary JSON:  {json_file}")
    print(f"  Fused images:  {fused_dir}")
    print("=" * 60)

    return all_metrics, avg_metrics


def main():
    parser = argparse.ArgumentParser(
        description="Test Adaptive Diffusion Fusion on M3FD test set"
    )
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Path to M3FD test data (default: data/m3fd/test)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: output/test_results)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to trained model checkpoint (default: checkpoints/best.pt)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device: cuda or cpu")
    parser.add_argument("--ddim_steps", type=int, default=50,
                        help="DDIM sampling steps (default: 50)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--start_idx", type=int, default=0,
                        help="Start index")
    parser.add_argument("--end_idx", type=int, default=None,
                        help="End index")
    parser.add_argument("--skip_fusion", action="store_true",
                        help="Skip fusion, only compute metrics on existing outputs")
    parser.add_argument("--adaptive", action="store_true",
                        help="Use adaptive diffusion checkpoint (adaptive_best.pt)")

    args = parser.parse_args()

    # Auto-select adaptive checkpoint if --adaptive flag is set
    checkpoint = args.checkpoint
    if args.adaptive and checkpoint is None:
        checkpoint = os.path.join(PROJECT_ROOT, "checkpoints", "adaptive_best.pt")

    run_testing(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        checkpoint=checkpoint,
        device=args.device,
        ddim_steps=args.ddim_steps,
        seed=args.seed,
        start_idx=args.start_idx,
        end_idx=args.end_idx,
        skip_fusion=args.skip_fusion,
    )


if __name__ == "__main__":
    main()
