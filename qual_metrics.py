"""
Standalone Quality Metrics Computation
Computes fusion quality metrics for a single IR/VIS/Fused image triplet.
Usage: python qual_metrics.py --ir path/to/ir.png --vis path/to/vis.png --fused path/to/fused.png
"""
import os
import sys
import argparse
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from scipy.stats import entropy


def mutual_information(a, b):
    """Compute Mutual Information between two grayscale images."""
    h, _, _ = np.histogram2d(a.ravel(), b.ravel(), bins=256)
    pxy = h / np.sum(h)
    px = np.sum(pxy, axis=1)
    py = np.sum(pxy, axis=0)
    px_py = px[:, None] * py[None, :]
    nz = pxy > 0
    return float(np.sum(pxy[nz] * np.log2(pxy[nz] / px_py[nz])))


def corr_coeff(a, b):
    """Compute Correlation Coefficient between two images."""
    return float(np.corrcoef(a.flatten(), b.flatten())[0, 1])


def entropy_val(img):
    """Compute Shannon Entropy of an image."""
    hist, _ = np.histogram(img.flatten(), bins=256, density=True)
    hist = hist[hist > 0]
    return float(entropy(hist))


def spatial_freq(img):
    """Compute Spatial Frequency of an image."""
    rf = np.diff(img.astype(np.float64), axis=0)
    cf = np.diff(img.astype(np.float64), axis=1)
    return float(np.sqrt(np.mean(rf**2) + np.mean(cf**2)))


def average_gradient(img):
    """Compute Average Gradient of an image."""
    gx, gy = np.gradient(img.astype(np.float64))
    return float(np.mean(np.sqrt(gx**2 + gy**2)))


def vif_metric(ref, dist):
    """Compute VIF (Visual Information Fidelity)."""
    try:
        from sewar.full_ref import vifp
        return float(vifp(ref, dist))
    except ImportError:
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


def compute_all_metrics(ir_img, vis_img, fus_img):
    """Compute all quality metrics for a fused image."""
    # Resize if shapes don't match (use fused as reference)
    h, w = fus_img.shape[:2]
    if ir_img.shape[:2] != (h, w):
        ir_img = cv2.resize(ir_img, (w, h))
    if vis_img.shape[:2] != (h, w):
        vis_img = cv2.resize(vis_img, (w, h))

    results = {
        "MI (IR,F)"   : mutual_information(ir_img, fus_img),
        "MI (VIS,F)"  : mutual_information(vis_img, fus_img),
        "SSIM (IR,F)" : ssim(ir_img, fus_img, data_range=255),
        "SSIM (VIS,F)": ssim(vis_img, fus_img, data_range=255),
        "CC (IR,F)"   : corr_coeff(ir_img, fus_img),
        "CC (VIS,F)"  : corr_coeff(vis_img, fus_img),
        "EN (F)"      : entropy_val(fus_img),
        "SD (F)"      : float(np.std(fus_img)),
        "SF (F)"      : spatial_freq(fus_img),
        "AG (F)"      : average_gradient(fus_img),
        "VIF (IR,F)"  : vif_metric(ir_img, fus_img),
        "VIF (VIS,F)" : vif_metric(vis_img, fus_img),
    }
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Compute quality metrics for IR-VIS image fusion"
    )
    parser.add_argument("--ir", type=str, required=True, help="Path to infrared image")
    parser.add_argument("--vis", type=str, required=True, help="Path to visible image")
    parser.add_argument("--fused", type=str, required=True, help="Path to fused image")
    args = parser.parse_args()

    # Load images
    ir_img = cv2.imread(args.ir, cv2.IMREAD_GRAYSCALE)
    vis_img = cv2.imread(args.vis, cv2.IMREAD_GRAYSCALE)
    fus_img = cv2.imread(args.fused, cv2.IMREAD_GRAYSCALE)

    for name, img, path in [("IR", ir_img, args.ir), ("VIS", vis_img, args.vis), ("FUSED", fus_img, args.fused)]:
        if img is None:
            print(f"Error: Failed to load {name} image: {path}")
            sys.exit(1)

    print("Images loaded:")
    print(f"  IR:    {args.ir} ({ir_img.shape})")
    print(f"  VIS:   {args.vis} ({vis_img.shape})")
    print(f"  FUSED: {args.fused} ({fus_img.shape})")

    results = compute_all_metrics(ir_img, vis_img, fus_img)

    print("\n===== QUALITY METRICS =====")
    for k, v in results.items():
        print(f"{k:18s}: {v:.4f}")


if __name__ == "__main__":
    main()
