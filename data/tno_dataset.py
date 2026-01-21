"""
TNO Dataset Loader for FusionINV
Handles loading and preprocessing of the TNO multiband image data collection
"""
import os
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader


class TNODataset(Dataset):
    """
    TNO Dataset for Infrared-Visible Image Fusion.

    Expected directory structure:
    tno/
    ├── vis/           # Visible images
    │   ├── image1.png
    │   ├── image2.png
    │   └── ...
    └── ir/            # Infrared images (same filenames)
        ├── image1.png
        ├── image2.png
        └── ...

    Or alternative structure:
    tno/
    ├── VIS/
    └── IR/
    """

    SUPPORTED_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}

    def __init__(
        self,
        root_dir: str,
        vis_subdir: str = "vis",
        ir_subdir: str = "ir",
        image_size: Tuple[int, int] = (512, 512),
        auto_detect_structure: bool = True
    ):
        """
        Initialize TNO Dataset.

        Args:
            root_dir: Path to TNO dataset root directory
            vis_subdir: Subdirectory name for visible images
            ir_subdir: Subdirectory name for infrared images
            image_size: Target size for images (width, height)
            auto_detect_structure: Auto-detect VIS/IR subdirectory names
        """
        self.root_dir = Path(root_dir)
        self.image_size = image_size

        # Auto-detect directory structure
        if auto_detect_structure:
            vis_subdir, ir_subdir = self._detect_structure()

        self.vis_dir = self.root_dir / vis_subdir
        self.ir_dir = self.root_dir / ir_subdir

        # Validate directories exist
        if not self.vis_dir.exists():
            raise FileNotFoundError(f"Visible images directory not found: {self.vis_dir}")
        if not self.ir_dir.exists():
            raise FileNotFoundError(f"Infrared images directory not found: {self.ir_dir}")

        # Find matching image pairs
        self.image_pairs = self._find_image_pairs()

        if len(self.image_pairs) == 0:
            raise ValueError(
                f"No matching image pairs found in {root_dir}. "
                f"Ensure visible and infrared images have matching filenames."
            )

        print(f"TNO Dataset loaded: {len(self.image_pairs)} image pairs")

    def _detect_structure(self) -> Tuple[str, str]:
        """
        Auto-detect the directory structure for visible and infrared images.

        Returns:
            Tuple of (vis_subdir, ir_subdir) names
        """
        subdirs = [d.name for d in self.root_dir.iterdir() if d.is_dir()]
        subdirs_lower = {d.lower(): d for d in subdirs}

        # Common naming patterns for visible images
        vis_patterns = ['vis', 'visible', 'rgb', 'vi']
        # Common naming patterns for infrared images
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
            vis_dir = "vis"
            print(f"Warning: Could not auto-detect visible directory, using '{vis_dir}'")
        if ir_dir is None:
            ir_dir = "ir"
            print(f"Warning: Could not auto-detect infrared directory, using '{ir_dir}'")

        return vis_dir, ir_dir

    def _find_image_pairs(self) -> List[Tuple[Path, Path]]:
        """
        Find matching image pairs between visible and infrared directories.

        Returns:
            List of (vis_path, ir_path) tuples
        """
        # Get all visible images
        vis_images = {}
        for ext in self.SUPPORTED_EXTENSIONS:
            for f in self.vis_dir.glob(f"*{ext}"):
                vis_images[f.stem.lower()] = f
            for f in self.vis_dir.glob(f"*{ext.upper()}"):
                vis_images[f.stem.lower()] = f

        # Get all infrared images
        ir_images = {}
        for ext in self.SUPPORTED_EXTENSIONS:
            for f in self.ir_dir.glob(f"*{ext}"):
                ir_images[f.stem.lower()] = f
            for f in self.ir_dir.glob(f"*{ext.upper()}"):
                ir_images[f.stem.lower()] = f

        # Find matching pairs
        pairs = []
        for name in vis_images:
            if name in ir_images:
                pairs.append((vis_images[name], ir_images[name]))

        # Sort by filename for reproducibility
        pairs.sort(key=lambda x: x[0].stem.lower())

        return pairs

    def __len__(self) -> int:
        """Return number of image pairs."""
        return len(self.image_pairs)

    def __getitem__(self, idx: int) -> Dict[str, any]:
        """
        Get an image pair.

        Args:
            idx: Index of the image pair

        Returns:
            Dictionary with 'vis', 'ir', 'vis_path', 'ir_path', 'name'
        """
        vis_path, ir_path = self.image_pairs[idx]

        # Load images
        vis_image = Image.open(vis_path)
        ir_image = Image.open(ir_path)

        # Convert to RGB if needed
        if vis_image.mode != 'RGB':
            vis_image = vis_image.convert('RGB')
        if ir_image.mode == 'L':
            ir_image = ir_image.convert('RGB')
        elif ir_image.mode != 'RGB':
            ir_image = ir_image.convert('RGB')

        # Resize
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
        """
        Get paths for a specific image pair.

        Args:
            idx: Index of the image pair

        Returns:
            Tuple of (vis_path, ir_path)
        """
        vis_path, ir_path = self.image_pairs[idx]
        return str(vis_path), str(ir_path)

    def get_all_names(self) -> List[str]:
        """Get list of all image pair names."""
        return [p[0].stem for p in self.image_pairs]


class TNODataLoader:
    """
    Convenience wrapper for loading TNO dataset for FusionINV.
    """

    def __init__(
        self,
        tno_root: str,
        image_size: Tuple[int, int] = (512, 512),
        vis_subdir: str = "vis",
        ir_subdir: str = "ir"
    ):
        """
        Initialize TNO data loader.

        Args:
            tno_root: Path to TNO dataset root
            image_size: Target image size
            vis_subdir: Visible images subdirectory
            ir_subdir: Infrared images subdirectory
        """
        self.dataset = TNODataset(
            root_dir=tno_root,
            vis_subdir=vis_subdir,
            ir_subdir=ir_subdir,
            image_size=image_size
        )
        self.current_idx = 0

    def __len__(self) -> int:
        return len(self.dataset)

    def __iter__(self):
        self.current_idx = 0
        return self

    def __next__(self) -> Dict:
        if self.current_idx >= len(self.dataset):
            raise StopIteration
        item = self.dataset[self.current_idx]
        self.current_idx += 1
        return item

    def get_pair(self, idx: int) -> Dict:
        """Get specific image pair by index."""
        return self.dataset[idx]

    def get_pair_by_name(self, name: str) -> Optional[Dict]:
        """Get image pair by name."""
        names = self.dataset.get_all_names()
        name_lower = name.lower()
        for i, n in enumerate(names):
            if n.lower() == name_lower:
                return self.dataset[i]
        return None


def download_tno_info():
    """Print information about obtaining the TNO dataset."""
    print("""
TNO Multiband Image Data Collection
=====================================

The TNO dataset is a benchmark for infrared-visible image fusion.

Download sources:
1. Official: https://figshare.com/articles/dataset/TNO_Image_Fusion_Dataset/1008029
2. GitHub mirrors with preprocessed versions

After downloading, organize the dataset as:
    data/tno/
    ├── vis/           # Visible images
    │   ├── image1.png
    │   └── ...
    └── ir/            # Infrared images
        ├── image1.png
        └── ...

Ensure visible and infrared images have matching filenames.
""")


if __name__ == "__main__":
    download_tno_info()
