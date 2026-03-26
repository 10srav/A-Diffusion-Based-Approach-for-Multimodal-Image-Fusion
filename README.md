# Adaptive Diffusion Fusion for Multimodal Image Fusion

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A custom **Adaptive Diffusion Fusion** model (~17M parameters) for infrared and visible image fusion. Trained from scratch on the M3FD dataset using a DDPM/DDIM diffusion framework with content-aware adaptive fusion attention.

Inspired by [FusionINV](https://ieeexplore.ieee.org/document/11114795) (IEEE TIP 2025).

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Adaptive Diffusion Fusion Pipeline                       │
│                                                                             │
│  ┌─────────────────────────┐    ┌─────────────────────────┐                │
│  │      Visible Image      │    │     Infrared Image      │                │
│  │       (RGB Input)       │    │    (Thermal Input)      │                │
│  └───────────┬─────────────┘    └───────────┬─────────────┘                │
│              │                              │                              │
│              ▼                              ▼                              │
│  ┌─────────────────────┐        ┌─────────────────────┐                   │
│  │   VIS Modality       │        │   IR Modality        │                   │
│  │   Encoder            │        │   Encoder            │                   │
│  │   (Multi-scale CNN)  │        │   (Multi-scale CNN)  │                   │
│  │   4 resolution lvls  │        │   4 resolution lvls  │                   │
│  └──┬──┬──┬──┬──────────┘        └──┬──┬──┬──┬──────────┘                   │
│     │  │  │  │                      │  │  │  │                              │
│     ▼  ▼  ▼  ▼                      ▼  ▼  ▼  ▼                              │
│  ┌──────────────────────────────────────────────────────────┐               │
│  │              ADAPTIVE FUSION (per-scale)                  │               │
│  │  ╔════════════════════════════════════════════════════╗   │               │
│  │  ║  Scale 0 (256x256): AdaptiveFusionGate            ║   │               │
│  │  ║  Scale 1 (128x128): AdaptiveFusionGate            ║   │               │
│  │  ║    → Conv gating: gate*IR + (1-gate)*VIS          ║   │               │
│  │  ╠════════════════════════════════════════════════════╣   │               │
│  │  ║  Scale 2 (64x64):  AdaptiveFusionAttention        ║   │               │
│  │  ║  Scale 3 (32x32):  AdaptiveFusionAttention        ║   │               │
│  │  ║    → Bidirectional cross-attention + gating        ║   │               │
│  │  ╚════════════════════════════════════════════════════╝   │               │
│  └──────────────────────┬───────────────────────────────────┘               │
│                         │ Multi-scale fused condition features               │
│                         ▼                                                    │
│  ┌──────────────────────────────────────────────────────────┐               │
│  │                 Denoising U-Net (~12.5M params)           │               │
│  │  ┌────────────────────────────────────────────────────┐  │               │
│  │  │ Encoder: [32, 64, 128, 256] channels               │  │               │
│  │  │ Self-attention at 64x64 and 32x32                  │  │               │
│  │  │ Sinusoidal timestep embeddings                     │  │               │
│  │  │ Bottleneck: 16x16 with self-attention              │  │               │
│  │  │ Decoder: Skip connections + condition injection    │  │               │
│  │  └────────────────────────────────────────────────────┘  │               │
│  │  Input: noisy image + scale-0 fused features             │               │
│  │  Output: predicted noise                                  │               │
│  └──────────────────────┬───────────────────────────────────┘               │
│                         │                                                    │
│                         ▼                                                    │
│              ┌────────────────────┐                                          │
│              │   DDIM Sampling    │                                          │
│              │   (50 steps)       │                                          │
│              │   + SDEdit from    │                                          │
│              │   pseudo GT        │                                          │
│              └──────────┬─────────┘                                          │
│                         │                                                    │
│                         ▼                                                    │
│              ┌────────────────────┐                                          │
│              │    FUSED IMAGE     │                                          │
│              │  (Visible-Style)   │                                          │
│              └────────────────────┘                                          │
│                                                                             │
│  TRAINING:                                                                  │
│  ├── Dataset: M3FD (200 train / 100 test pairs)                             │
│  ├── Pseudo GT: Visible-dominant blend (vis_weight=0.7)                     │
│  ├── Loss: DDPM MSE with min-SNR weighting (gamma=5.0)                      │
│  ├── Optimizer: AdamW (lr=2e-4, weight_decay=1e-4)                          │
│  ├── Schedule: Linear warmup + cosine decay                                 │
│  ├── EMA: decay=0.9999                                                      │
│  └── AMP: Mixed precision on CUDA                                           │
│                                                                             │
│  INFERENCE:                                                                 │
│  ├── SDEdit: Noise pseudo GT, then denoise with trained model               │
│  ├── DDIM: 50 deterministic steps (eta=0)                                   │
│  ├── Strength: 0.95 (start from high noise level)                           │
│  └── Image size: 256x256                                                    │
│                                                                             │
│  KEY PARAMETERS:                                                            │
│  ├── Total model params: ~17M                                               │
│  ├── Diffusion timesteps: 1000 (linear beta schedule)                       │
│  ├── DDIM sampling steps: 50                                                │
│  └── Feature channels: [16, 32, 64, 128] (encoders)                        │
│                         [32, 64, 128, 256] (U-Net)                          │
└─────────────────────────────────────────────────────────────────────────────┘
```

### How It Works

1. **Dual Modality Encoding**: Separate multi-scale CNN encoders extract features from IR and VIS images at 4 resolution levels
2. **Adaptive Fusion**: Content-aware fusion merges IR and VIS features:
   - **High-res (256x256, 128x128)**: Convolutional gating — learns spatially-varying blend weights
   - **Low-res (64x64, 32x32)**: Bidirectional cross-attention + gating — IR attends to VIS and vice versa
3. **Conditioned Denoising**: The fused multi-scale features condition a U-Net denoiser via skip-connection injection
4. **SDEdit Refinement**: A pseudo ground-truth (visible-dominant blend) is partially noised, then refined by the diffusion model
5. **Output**: Visible-style fused image preserving natural colors with enhanced IR detail

## Features

- **Custom Lightweight Model**: ~17M parameters, trainable on a single GPU
- **Adaptive Fusion Attention**: Content-aware gating + cross-attention at multiple scales
- **SDEdit Inference**: Refines a pseudo GT for cleaner outputs than pure noise generation
- **DDIM Fast Sampling**: 50-step deterministic sampling
- **Min-SNR Loss Weighting**: Balanced training across noise levels
- **EMA Weights**: Exponential moving average for stable inference
- **M3FD Dataset Support**: Auto-detection, train/test splitting, data augmentation

## Installation

```bash
# Create conda environment
conda create -n fusioninv python=3.9
conda activate fusioninv

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Complete Pipeline (Setup + Train + Test)
```bash
python run_all.py --source_dir /path/to/M3FD --epochs 100
```

### Step-by-Step

#### 1. Setup M3FD Dataset
```bash
# From local M3FD folder
python setup_m3fd.py --source_dir /path/to/M3FD

# Or download from Google Drive
python setup_m3fd.py --download
```

#### 2. Train
```bash
python run_train.py --epochs 500 --batch_size 4 --lr 2e-4
```

#### 3. Test
```bash
python run_test.py --checkpoint checkpoints/best.pt
```

### Single Image Pair Inference
```bash
python fusioninv.py \
    --checkpoint checkpoints/best.pt \
    --vis_image_path ./data/sample_vis.png \
    --ir_image_path ./data/sample_ir.png \
    --output_path ./output
```

### Batch Processing (M3FD)
```bash
python fusioninv.py \
    --checkpoint checkpoints/best.pt \
    --data_dir ./data/m3fd/test \
    --output_path ./output/results
```

### TNO Dataset Processing
```bash
python process_tno.py \
    --tno_root ./data/tno \
    --output_dir ./output/tno_results \
    --checkpoint checkpoints/best.pt
```

## M3FD Dataset Setup

1. Download M3FD dataset from [Google Drive](https://drive.google.com/drive/folders/1H-oO7bgRuVFYDcMGvxstT1nmy0WF_Y_6)
2. Run setup to split into train/test:
```bash
python setup_m3fd.py --source_dir /path/to/M3FD
```

Expected structure after setup:
```
data/m3fd/
├── train/
│   ├── vis/    # 200 visible images
│   └── ir/     # 200 infrared images
├── test/
│   ├── vis/    # 100 visible images
│   └── ir/     # 100 infrared images
└── split_info.txt
```

The loader auto-detects common directory names: `vis`/`Vis`/`visible`/`RGB` and `ir`/`Ir`/`infrared`/`thermal`.

## Project Structure

```
AdaptiveDiffusionFusion/
├── models/
│   ├── __init__.py              # Module exports
│   ├── adaptive_unet.py         # Denoising U-Net (~12.5M params)
│   ├── adaptive_diffusion.py    # DDPM/DDIM diffusion process
│   └── adaptive_fusion_net.py   # Full model: encoders + fusion + U-Net
├── data/
│   ├── m3fd_dataset.py          # M3FD dataset loader + train/test split
│   └── tno_dataset.py           # TNO dataset loader
├── checkpoints/                 # Trained model weights
├── output/                      # Generated outputs
├── fusioninv.py                 # Main inference pipeline
├── process_tno.py               # TNO batch processing script
├── run_train.py                 # Training script
├── run_test.py                  # Testing + metrics script
├── run_all.py                   # Complete pipeline (setup + train + test)
├── setup_m3fd.py                # M3FD dataset download + split
├── qual_metrics.py              # Standalone quality metrics computation
├── create_pdf_report.py         # PDF report generation
├── requirements.txt
└── README.md
```

## Parameters

### Training Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--epochs` | 500 | Training epochs |
| `--batch_size` | 4 | Batch size |
| `--lr` | 2e-4 | Learning rate |
| `--seed` | 42 | Random seed |
| `--save_every` | 10 | Save checkpoint every N epochs |
| `--sample_every` | 10 | Generate samples every N epochs |

### Inference Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--checkpoint` | checkpoints/best.pt | Path to trained model checkpoint |
| `--ddim_steps` | 50 | DDIM sampling steps |
| `--device` | cuda | Device (cuda or cpu) |
| `--seed` | 42 | Random seed |

### Model Architecture
| Component | Channels | Description |
|-----------|----------|-------------|
| Modality Encoders | [16, 32, 64, 128] | Multi-scale feature extraction |
| U-Net | [32, 64, 128, 256] | Denoising backbone |
| Timestep Embedding | 256-dim | Sinusoidal + MLP |
| Diffusion | 1000 steps | Linear beta schedule |

## Quality Metrics

The testing pipeline (`run_test.py`) computes these metrics:

| Metric | Description |
|--------|-------------|
| MI (IR/VIS, F) | Mutual Information between source and fused |
| SSIM (IR/VIS, F) | Structural Similarity Index |
| CC (IR/VIS, F) | Correlation Coefficient |
| EN | Shannon Entropy of fused image |
| SD | Standard Deviation of fused image |
| SF | Spatial Frequency |
| AG | Average Gradient |
| VIF (IR/VIS, F) | Visual Information Fidelity |

## Citation

This implementation is inspired by:

```bibtex
@article{liang2025fusioninv,
  title={FusionINV: A Diffusion-Based Approach for Multimodal Image Fusion},
  author={Liang, Pengwei and Jiang, Junjun and Ma, Qing and Wang, Chenyang and Liu, Xianming and Ma, Jiayi},
  journal={IEEE Transactions on Image Processing},
  volume={34},
  pages={5355--5368},
  year={2025}
}
```

**Note**: This is a re-implementation using a custom lightweight architecture (~17M params), not the original FusionINV which uses Stable Diffusion v1.5 (~1.1B params) in a training-free manner.

## Acknowledgements

- [Original FusionINV Paper](https://ieeexplore.ieee.org/document/11114795)
- [Official FusionINV Implementation](https://github.com/erfect2020/FusionINV)
- [M3FD Dataset](https://drive.google.com/drive/folders/1H-oO7bgRuVFYDcMGvxstT1nmy0WF_Y_6)

## License

This project is licensed under the MIT License.
