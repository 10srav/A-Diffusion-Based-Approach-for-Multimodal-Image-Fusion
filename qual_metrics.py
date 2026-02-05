import os
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from sewar.full_ref import vifp
from scipy.stats import entropy

# -------- FOLDERS --------
# Using project's existing images
BASE = os.path.dirname(os.path.abspath(__file__))
ir_path  = os.path.join(BASE, "data", "sample_ir.png")
vis_path = os.path.join(BASE, "data", "sample_vis.png")
fus_path = os.path.join(BASE, "output", "fused_demo.png")

# -------- LOAD IMAGES --------
ir_img  = cv2.imread(ir_path, cv2.IMREAD_GRAYSCALE)
vis_img = cv2.imread(vis_path, cv2.IMREAD_GRAYSCALE)
fus_img = cv2.imread(fus_path, cv2.IMREAD_GRAYSCALE)

for name, img, path in [("IR", ir_img, ir_path), ("VIS", vis_img, vis_path), ("FUSED", fus_img, fus_path)]:
    if img is None:
        raise RuntimeError(f"Failed to load {name} image: {path}")

print("Using images:")
print(f"  IR:    {ir_path}")
print(f"  VIS:   {vis_path}")
print(f"  FUSED: {fus_path}")
print(f"  Shapes: IR={ir_img.shape}, VIS={vis_img.shape}, FUSED={fus_img.shape}")

# Resize if shapes don't match (use fused as reference)
if ir_img.shape != fus_img.shape:
    ir_img = cv2.resize(ir_img, (fus_img.shape[1], fus_img.shape[0]))
    print(f"  Resized IR to {ir_img.shape}")
if vis_img.shape != fus_img.shape:
    vis_img = cv2.resize(vis_img, (fus_img.shape[1], fus_img.shape[0]))
    print(f"  Resized VIS to {vis_img.shape}")

# -------- METRICS --------
def mutual_information(a, b):
    h, _, _ = np.histogram2d(a.ravel(), b.ravel(), bins=256)
    pxy = h / np.sum(h)
    px = np.sum(pxy, axis=1)
    py = np.sum(pxy, axis=0)
    px_py = px[:, None] * py[None, :]
    nz = pxy > 0
    return np.sum(pxy[nz] * np.log2(pxy[nz] / px_py[nz]))

def corr_coeff(a, b):
    return np.corrcoef(a.flatten(), b.flatten())[0, 1]

def entropy_val(img):
    hist, _ = np.histogram(img.flatten(), bins=256, density=True)
    hist = hist[hist > 0]
    return entropy(hist)

def spatial_freq(img):
    rf = np.diff(img, axis=0)
    cf = np.diff(img, axis=1)
    return np.sqrt(np.mean(rf**2) + np.mean(cf**2))

def average_gradient(img):
    gx, gy = np.gradient(img.astype(np.float32))
    return np.mean(np.sqrt(gx**2 + gy**2))

# -------- COMPUTE --------
results = {
    "MI (IR,F)"   : mutual_information(ir_img, fus_img),
    "MI (VIS,F)"  : mutual_information(vis_img, fus_img),
    "SSIM (IR,F)" : ssim(ir_img, fus_img, data_range=255),
    "SSIM (VIS,F)": ssim(vis_img, fus_img, data_range=255),
    "CC (IR,F)"   : corr_coeff(ir_img, fus_img),
    "CC (VIS,F)"  : corr_coeff(vis_img, fus_img),
    "EN (F)"      : entropy_val(fus_img),
    "SD (F)"      : np.std(fus_img),
    "SF (F)"      : spatial_freq(fus_img),
    "AG (F)"      : average_gradient(fus_img),
    "VIF (IR,F)"  : vifp(ir_img, fus_img),
    "VIF (VIS,F)" : vifp(vis_img, fus_img),
}

print("\n===== QUALITATIVE METRICS (ONE IMAGE) =====")
for k, v in results.items():
    print(f"{k:18s}: {v:.4f}")
