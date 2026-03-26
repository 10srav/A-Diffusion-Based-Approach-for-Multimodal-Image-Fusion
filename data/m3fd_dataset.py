"""
M3FD Dataset Loader for Adaptive Diffusion Fusion
Handles loading and preprocessing of M3FD (Multi-spectral) infrared-visible image pairs.
Supports train/test split: 200 images for training, 100 images for testing.
"""
import os
import random
import shutil
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF


class M3FDDataset(Dataset):
    """
    M3FD Dataset for Infrared-Visible Image Fusion.

    Expected directory structure (auto-detected):
    M3FD/
    ├── Ir/  (or ir/)
    │   ├── 00001.png
    │   └── ...
    └── Vis/ (or vi/ or vis/)
        ├── 00001.png
        └── ...
    """

    SUPPORTED_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}

    def __init__(
        self,
        root_dir: str,
        vis_subdir: str = "Vis",
        ir_subdir: str = "Ir",
        image_size: Tuple[int, int] = (512, 512),
        auto_detect_structure: bool = True
    ):
        self.root_dir = Path(root_dir)
        self.image_size = image_size

        if auto_detect_structure:
            vis_subdir, ir_subdir = self._detect_structure()

        self.vis_dir = self.root_dir / vis_subdir
        self.ir_dir = self.root_dir / ir_subdir

        if not self.vis_dir.exists():
            raise FileNotFoundError(f"Visible images directory not found: {self.vis_dir}")
        if not self.ir_dir.exists():
            raise FileNotFoundError(f"Infrared images directory not found: {self.ir_dir}")

        self.image_pairs = self._find_image_pairs()

        if len(self.image_pairs) == 0:
            raise ValueError(
                f"No matching image pairs found in {root_dir}. "
                f"Ensure visible and infrared images have matching filenames."
            )

        print(f"M3FD Dataset loaded: {len(self.image_pairs)} image pairs")

    def _detect_structure(self) -> Tuple[str, str]:
        """Auto-detect visible/infrared subdirectory names."""
        subdirs = [d.name for d in self.root_dir.iterdir() if d.is_dir()]
        subdirs_lower = {d.lower(): d for d in subdirs}

        vis_patterns = ['vis', 'visible', 'rgb', 'vi']
        ir_patterns = ['ir', 'infrared', 'thermal', 'lwir', 'nir']

        vis_dir = None
        ir_dir = None

        for pattern in vis_patterns:
            if pattern in subdirs_lower:
                vis_dir = subdirs_lower[pattern]
                break

        for pattern in ir_patterns:
            if pattern in subdirs_lower:
                ir_dir = subdirs_lower[pattern]
                break

        if vis_dir is None:
            vis_dir = "Vis"
            print(f"Warning: Could not auto-detect visible directory, using '{vis_dir}'")
        if ir_dir is None:
            ir_dir = "Ir"
            print(f"Warning: Could not auto-detect infrared directory, using '{ir_dir}'")

        return vis_dir, ir_dir

    def _find_image_pairs(self) -> List[Tuple[Path, Path]]:
        """Find matching image pairs between visible and infrared directories."""
        vis_images = {}
        for ext in self.SUPPORTED_EXTENSIONS:
            for f in self.vis_dir.glob(f"*{ext}"):
                vis_images[f.stem.lower()] = f
            for f in self.vis_dir.glob(f"*{ext.upper()}"):
                vis_images[f.stem.lower()] = f

        ir_images = {}
        for ext in self.SUPPORTED_EXTENSIONS:
            for f in self.ir_dir.glob(f"*{ext}"):
                ir_images[f.stem.lower()] = f
            for f in self.ir_dir.glob(f"*{ext.upper()}"):
                ir_images[f.stem.lower()] = f

        pairs = []
        for name in vis_images:
            if name in ir_images:
                pairs.append((vis_images[name], ir_images[name]))

        pairs.sort(key=lambda x: x[0].stem.lower())
        return pairs

    def __len__(self) -> int:
        return len(self.image_pairs)

    def __getitem__(self, idx: int) -> Dict:
        vis_path, ir_path = self.image_pairs[idx]

        vis_image = Image.open(vis_path)
        ir_image = Image.open(ir_path)

        if vis_image.mode != 'RGB':
            vis_image = vis_image.convert('RGB')
        if ir_image.mode == 'L':
            ir_image = ir_image.convert('RGB')
        elif ir_image.mode != 'RGB':
            ir_image = ir_image.convert('RGB')

        vis_image = vis_image.resize(self.image_size, Image.LANCZOS)
        ir_image = ir_image.resize(self.image_size, Image.LANCZOS)

        return {
            'vis': vis_image,
            'ir': ir_image,
            'vis_path': str(vis_path),
            'ir_path': str(ir_path),
            'name': vis_path.stem
        }

    def get_pair_paths(self, idx: int) -> Tuple[str, str]:
        vis_path, ir_path = self.image_pairs[idx]
        return str(vis_path), str(ir_path)

    def get_all_names(self) -> List[str]:
        return [p[0].stem for p in self.image_pairs]


class M3FDTrainDataset(Dataset):
    """
    M3FD Dataset for training the Adaptive Diffusion Model.
    Returns tensors at 256x256 with data augmentation and pseudo GT.
    """

    SUPPORTED_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}

    def __init__(
        self,
        root_dir: str,
        image_size: Tuple[int, int] = (256, 256),
        augment: bool = True,
    ):
        self.root_dir = Path(root_dir)
        self.image_size = image_size
        self.augment = augment

        # Auto-detect structure
        subdirs = [d.name for d in self.root_dir.iterdir() if d.is_dir()]
        subdirs_lower = {d.lower(): d for d in subdirs}

        vis_name = None
        ir_name = None
        for p in ['vis', 'visible', 'rgb', 'vi']:
            if p in subdirs_lower:
                vis_name = subdirs_lower[p]
                break
        for p in ['ir', 'infrared', 'thermal', 'lwir', 'nir']:
            if p in subdirs_lower:
                ir_name = subdirs_lower[p]
                break

        self.vis_dir = self.root_dir / (vis_name or "vis")
        self.ir_dir = self.root_dir / (ir_name or "ir")

        self.image_pairs = self._find_image_pairs()
        print(f"M3FD Train Dataset: {len(self.image_pairs)} pairs (augment={augment})")

    def _find_image_pairs(self) -> List[Tuple[Path, Path]]:
        vis_images = {}
        for ext in self.SUPPORTED_EXTENSIONS:
            for f in self.vis_dir.glob(f"*{ext}"):
                vis_images[f.stem.lower()] = f
            for f in self.vis_dir.glob(f"*{ext.upper()}"):
                vis_images[f.stem.lower()] = f

        ir_images = {}
        for ext in self.SUPPORTED_EXTENSIONS:
            for f in self.ir_dir.glob(f"*{ext}"):
                ir_images[f.stem.lower()] = f
            for f in self.ir_dir.glob(f"*{ext.upper()}"):
                ir_images[f.stem.lower()] = f

        pairs = [(vis_images[n], ir_images[n]) for n in vis_images if n in ir_images]
        pairs.sort(key=lambda x: x[0].stem.lower())
        return pairs

    def __len__(self) -> int:
        return len(self.image_pairs)

    def __getitem__(self, idx: int) -> Dict:
        vis_path, ir_path = self.image_pairs[idx]

        vis = Image.open(vis_path).convert('RGB')
        ir = Image.open(ir_path).convert('RGB')

        if self.augment:
            # Random horizontal flip (same for both)
            if random.random() > 0.5:
                vis = TF.hflip(vis)
                ir = TF.hflip(ir)

            # Random vertical flip
            if random.random() > 0.5:
                vis = TF.vflip(vis)
                ir = TF.vflip(ir)

            # Random rotation (same angle)
            angle = random.uniform(-10, 10)
            vis = TF.rotate(vis, angle)
            ir = TF.rotate(ir, angle)

            # Random crop (resize larger, then crop)
            resize_size = int(self.image_size[0] * 1.125)  # 288
            vis = TF.resize(vis, [resize_size, resize_size])
            ir = TF.resize(ir, [resize_size, resize_size])
            i, j, h, w = transforms.RandomCrop.get_params(
                vis, output_size=self.image_size
            )
            vis = TF.crop(vis, i, j, h, w)
            ir = TF.crop(ir, i, j, h, w)
        else:
            vis = TF.resize(vis, list(self.image_size))
            ir = TF.resize(ir, list(self.image_size))

        # To tensor [-1, 1]
        vis_tensor = TF.normalize(TF.to_tensor(vis), [0.5] * 3, [0.5] * 3)
        ir_tensor = TF.normalize(TF.to_tensor(ir), [0.5] * 3, [0.5] * 3)

        # Create pseudo ground truth
        gt_fused = self._create_pseudo_gt(ir_tensor, vis_tensor)

        return {
            'vis': vis_tensor,
            'ir': ir_tensor,
            'gt_fused': gt_fused,
            'name': vis_path.stem
        }

    @staticmethod
    def _create_pseudo_gt(ir_tensor, vis_tensor, vis_weight=0.7):
        """Visible-dominant fusion: visible base + subtle IR enhancement."""
        ir_01 = (ir_tensor + 1) / 2
        vis_01 = (vis_tensor + 1) / 2

        ir_y = 0.299 * ir_01[0:1] + 0.587 * ir_01[1:2] + 0.114 * ir_01[2:3]
        vis_y = 0.299 * vis_01[0:1] + 0.587 * vis_01[1:2] + 0.114 * vis_01[2:3]

        # Visible-biased: use visible as base, add IR only in dark regions
        ir_boost = torch.relu(ir_y - vis_y)
        fused_y = vis_y + (1.0 - vis_weight) * ir_boost

        vis_cb = vis_01[2:3] - vis_y
        vis_cr = vis_01[0:1] - vis_y

        fused_r = fused_y + vis_cr
        fused_g = fused_y - 0.344136 * vis_cb - 0.714136 * vis_cr
        fused_b = fused_y + vis_cb

        fused = torch.cat([fused_r, fused_g, fused_b], dim=0)
        fused = torch.clamp(fused, 0, 1)
        return fused * 2 - 1


def split_m3fd_dataset(
    source_dir: str,
    output_dir: str,
    num_train: int = 200,
    num_test: int = 100,
    seed: int = 42
):
    """
    Split M3FD dataset into train and test sets.

    Args:
        source_dir: Path to original M3FD dataset (with Ir/ and Vis/ subdirs)
        output_dir: Path to output directory
        num_train: Number of training images (default: 200)
        num_test: Number of test images (default: 100)
        seed: Random seed for reproducible split
    """
    source_path = Path(source_dir)
    output_path = Path(output_dir)

    # Auto-detect structure
    subdirs = [d.name for d in source_path.iterdir() if d.is_dir()]
    subdirs_lower = {d.lower(): d for d in subdirs}

    vis_name = None
    ir_name = None
    for p in ['vis', 'visible', 'rgb', 'vi']:
        if p in subdirs_lower:
            vis_name = subdirs_lower[p]
            break
    for p in ['ir', 'infrared', 'thermal', 'lwir', 'nir']:
        if p in subdirs_lower:
            ir_name = subdirs_lower[p]
            break

    if vis_name is None or ir_name is None:
        raise FileNotFoundError(
            f"Could not find Vis/Ir directories in {source_dir}. "
            f"Found: {subdirs}"
        )

    vis_dir = source_path / vis_name
    ir_dir = source_path / ir_name

    # Find matching pairs
    supported_ext = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
    vis_files = {}
    for ext in supported_ext:
        for f in vis_dir.glob(f"*{ext}"):
            vis_files[f.stem.lower()] = f
        for f in vis_dir.glob(f"*{ext.upper()}"):
            vis_files[f.stem.lower()] = f

    ir_files = {}
    for ext in supported_ext:
        for f in ir_dir.glob(f"*{ext}"):
            ir_files[f.stem.lower()] = f
        for f in ir_dir.glob(f"*{ext.upper()}"):
            ir_files[f.stem.lower()] = f

    # Find common names
    common_names = sorted(set(vis_files.keys()) & set(ir_files.keys()))
    total = len(common_names)
    print(f"Found {total} matching image pairs in M3FD dataset")

    if total < num_train + num_test:
        print(f"Warning: Only {total} pairs available. Adjusting split...")
        if total >= num_test:
            num_train = total - num_test
        else:
            num_train = int(total * 0.67)
            num_test = total - num_train
        print(f"  Adjusted: {num_train} train, {num_test} test")

    # Shuffle and split
    random.seed(seed)
    shuffled = common_names.copy()
    random.shuffle(shuffled)

    train_names = shuffled[:num_train]
    test_names = shuffled[num_train:num_train + num_test]

    # Create output directories
    for split in ['train', 'test']:
        for modality in ['vis', 'ir']:
            (output_path / split / modality).mkdir(parents=True, exist_ok=True)

    # Copy files
    print(f"\nCopying {num_train} training pairs...")
    for name in train_names:
        shutil.copy2(vis_files[name], output_path / "train" / "vis" / vis_files[name].name)
        shutil.copy2(ir_files[name], output_path / "train" / "ir" / ir_files[name].name)

    print(f"Copying {num_test} testing pairs...")
    for name in test_names:
        shutil.copy2(vis_files[name], output_path / "test" / "vis" / vis_files[name].name)
        shutil.copy2(ir_files[name], output_path / "test" / "ir" / ir_files[name].name)

    # Save split info
    with open(output_path / "split_info.txt", 'w') as f:
        f.write(f"M3FD Dataset Split\n")
        f.write(f"==================\n")
        f.write(f"Source: {source_dir}\n")
        f.write(f"Seed: {seed}\n")
        f.write(f"Train: {num_train} pairs\n")
        f.write(f"Test: {num_test} pairs\n\n")
        f.write(f"Train images:\n")
        for name in sorted(train_names):
            f.write(f"  {name}\n")
        f.write(f"\nTest images:\n")
        for name in sorted(test_names):
            f.write(f"  {name}\n")

    print(f"\nSplit complete!")
    print(f"  Train: {output_path / 'train'} ({num_train} pairs)")
    print(f"  Test:  {output_path / 'test'} ({num_test} pairs)")
    print(f"  Info:  {output_path / 'split_info.txt'}")

    return train_names, test_names


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Split M3FD dataset into train/test")
    parser.add_argument("--source", type=str, required=True, help="Path to M3FD dataset")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--num_train", type=int, default=200, help="Number of training images")
    parser.add_argument("--num_test", type=int, default=100, help="Number of test images")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    split_m3fd_dataset(args.source, args.output, args.num_train, args.num_test, args.seed)
