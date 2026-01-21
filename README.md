# FusionINV: A Diffusion-Based Approach for Multimodal Image Fusion

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Implementation of **FusionINV** (IEEE TIP 2025) - a training-free method for infrared and visible image fusion using Stable Diffusion v1.5 inversion.

## Architecture

```
                            FusionINV Pipeline
    ┌─────────────────────────────────────────────────────────────────┐
    │                                                                 │
    │   ┌─────────────┐         ┌─────────────┐                       │
    │   │  Visible    │         │  Infrared   │                       │
    │   │   Image     │         │   Image     │                       │
    │   └──────┬──────┘         └──────┬──────┘                       │
    │          │                       │                              │
    │          ▼                       ▼                              │
    │   ┌─────────────────────────────────────┐                       │
    │   │          VAE Encoder                │                       │
    │   │    (Image → Latent Space)           │                       │
    │   └──────┬───────────────────────┬──────┘                       │
    │          │                       │                              │
    │          ▼                       ▼                              │
    │   ┌─────────────┐         ┌─────────────────────────┐           │
    │   │   DDIM      │         │    DDIM Inversion       │           │
    │   │ Inversion   │────────▶│  + Visible Cues (λ)     │           │
    │   │  (VIS)      │         │       (IR)              │           │
    │   └──────┬──────┘         └───────────┬─────────────┘           │
    │          │                            │                         │
    │          │    ┌────────────────┐      │                         │
    │          └───▶│ Feature Memory │◀─────┘                         │
    │               │ (z_vis, z_inf) │                                │
    │               └───────┬────────┘                                │
    │                       │                                         │
    │                       ▼                                         │
    │   ┌─────────────────────────────────────────────────────────┐   │
    │   │              Fusion Generation (DDIM)                   │   │
    │   │  ┌─────────────────────────────────────────────────┐    │   │
    │   │  │  Stage 1 (t > T1): IR Feature Injection         │    │   │
    │   │  │  Stage 2 (T2 < t ≤ T1): VIS Feature Refinement  │    │   │
    │   │  │  Stage 3 (t ≤ T2): CFG Denoising                │    │   │
    │   │  └─────────────────────────────────────────────────┘    │   │
    │   └─────────────────────────┬───────────────────────────────┘   │
    │                             │                                   │
    │                             ▼                                   │
    │                   ┌─────────────────┐                           │
    │                   │   VAE Decoder   │                           │
    │                   └────────┬────────┘                           │
    │                            │                                    │
    │                            ▼                                    │
    │                   ┌─────────────────┐                           │
    │                   │  Fused Image    │                           │
    │                   │ (Visible-Style) │                           │
    │                   └─────────────────┘                           │
    │                                                                 │
    └─────────────────────────────────────────────────────────────────┘

    Key Components:
    ├── VAE Encoder/Decoder: Stable Diffusion v1.5 VAE
    ├── U-Net: Pre-trained SD v1.5 denoiser (frozen)
    ├── CLIP Text Encoder: Optional text guidance
    └── Feature Memory: Stores noise predictions for fusion
```

### How It Works

1. **Visible Inversion**: The visible image is inverted through DDIM to capture its noise trajectory `z_vis`
2. **Infrared Inversion with Visible Cues**: The infrared image is inverted with guidance from visible features (controlled by `λ`)
3. **Multi-Stage Fusion**: During denoising:
   - **Early Stage (t > T1=70)**: Inject IR structural features
   - **Middle Stage (T2=40 < t ≤ T1)**: Refine with VIS appearance
   - **Late Stage (t ≤ T2)**: Standard CFG denoising for detail enhancement

## Features

- **Training-Free**: Uses pre-trained Stable Diffusion v1.5 directly
- **Visible-Style Output**: Produces fused images compatible with foundation models (SAM, Grounding DINO)
- **Text-Interactive**: Supports text-guided fusion control
- **Fast Inference**: ~21s on RTX 3090 with BF16 precision

## 🔧 Installation

```bash
# Create conda environment
conda create -n fusioninv python=3.9
conda activate fusioninv

# Install dependencies
pip install -r requirements.txt
```

## 🚀 Quick Start

### Single Image Pair
```bash
python fusioninv.py \
    --vis_image_path ./data/sample_vis.png \
    --ir_image_path ./data/sample_ir.png \
    --output_path ./output \
    --seed 42
```

### TNO Dataset Batch Processing
```bash
# Process entire TNO dataset
python fusioninv.py \
    --tno_root ./data/tno \
    --output_path ./output/tno_results \
    --seed 42

# Process specific range
python fusioninv.py \
    --tno_root ./data/tno \
    --output_path ./output/tno_results \
    --start_idx 0 \
    --end_idx 10 \
    --seed 42
```

### Alternative: Dedicated TNO Script
```bash
python process_tno.py \
    --tno_root ./data/tno \
    --output_dir ./output/tno_results
```

### Streamlit Demo
```bash
streamlit run app.py
```

## 📥 TNO Dataset Setup

1. Download TNO dataset from [Figshare](https://figshare.com/articles/dataset/TNO_Image_Fusion_Dataset/1008029)
2. Organize the dataset:
```
data/tno/
├── vis/           # Visible images
│   ├── image1.png
│   ├── image2.png
│   └── ...
└── ir/            # Infrared images (matching filenames)
    ├── image1.png
    ├── image2.png
    └── ...
```

The loader auto-detects common directory names: `vis`/`VIS`/`visible`/`RGB` and `ir`/`IR`/`infrared`/`thermal`.

## 📂 Project Structure

```
FusionINV/
├── models/
│   ├── vae_encoder.py        # VAE latent space conversion
│   ├── inversion.py          # DDPM inversion with visible cues
│   ├── attention_hooks.py    # Self-attention K,V extraction
│   └── fusion.py             # Appearance injection module
├── data/
│   ├── tno_dataset.py        # TNO dataset loader
│   └── tno/                  # TNO dataset (user-provided)
├── output/                   # Generated outputs
├── fusioninv.py              # Main pipeline
├── process_tno.py            # TNO batch processing script
├── app.py                    # Streamlit demo
├── requirements.txt
└── README.md
```

## ⚙️ Parameters

### Fusion Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--lambda_vis` | 0.08 | Visible cues guidance strength |
| `--num_steps` | 80 | Total diffusion steps |
| `--t1` | 70 | IR injection cutoff step |
| `--t2` | 40 | VIS refinement cutoff step |
| `--guidance_scale` | 7.5 | Classifier-free guidance scale |

### TNO Dataset Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--tno_root` | None | Path to TNO dataset root |
| `--vis_subdir` | vis | Visible images subdirectory |
| `--ir_subdir` | ir | Infrared images subdirectory |
| `--start_idx` | 0 | Start index for batch processing |
| `--end_idx` | None | End index (None for all images) |

## 📌 Citation

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

## 🙏 Acknowledgements

- [Stable Diffusion v1.5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5)
- [Original FusionINV Paper](https://ieeexplore.ieee.org/document/11114795)
- [Official Implementation](https://github.com/erfect2020/FusionINV)

## 📄 License

This project is licensed under the MIT License.
