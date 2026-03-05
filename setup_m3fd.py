"""
M3FD Dataset Setup Script
Downloads M3FD dataset from Google Drive and splits into train/test sets.
Train: 200 images, Test: 100 images
"""
import os
import sys
import zipfile
import argparse
from pathlib import Path

# Try to import gdown for Google Drive downloads
try:
    import gdown
    HAS_GDOWN = True
except ImportError:
    HAS_GDOWN = False


# Google Drive folder/file ID
GDRIVE_FOLDER_URL = "https://drive.google.com/drive/folders/1H-oO7bgRuVFYDcMGvxstT1nmy0WF_Y_6"


def download_m3fd(output_dir: str):
    """
    Download M3FD dataset from Google Drive.

    Args:
        output_dir: Directory to download the dataset to
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if not HAS_GDOWN:
        print("=" * 60)
        print("ERROR: 'gdown' package not installed.")
        print("Install it with: pip install gdown")
        print("=" * 60)
        sys.exit(1)

    print("=" * 60)
    print("Downloading M3FD dataset from Google Drive...")
    print(f"Source: {GDRIVE_FOLDER_URL}")
    print(f"Target: {output_dir}")
    print("=" * 60)

    try:
        gdown.download_folder(
            url=GDRIVE_FOLDER_URL,
            output=str(output_path),
            quiet=False
        )
        print("\nDownload complete!")
    except Exception as e:
        print(f"\nAutomatic download failed: {e}")
        print("\n" + "=" * 60)
        print("MANUAL DOWNLOAD INSTRUCTIONS:")
        print("=" * 60)
        print(f"1. Open this URL in your browser:")
        print(f"   {GDRIVE_FOLDER_URL}")
        print(f"2. Download all files/folders")
        print(f"3. Extract to: {output_dir}")
        print(f"4. Ensure the structure is:")
        print(f"   {output_dir}/")
        print(f"   ├── Ir/  (or ir/)")
        print(f"   │   ├── 00001.png")
        print(f"   │   └── ...")
        print(f"   └── Vis/ (or vi/ or vis/)")
        print(f"       ├── 00001.png")
        print(f"       └── ...")
        print("=" * 60)
        sys.exit(1)

    # Check for nested directories or zip files
    _handle_downloaded_content(output_path)


def _handle_downloaded_content(download_path: Path):
    """Handle extracted content - check for zips or nested folders."""
    # Check for zip files and extract
    for zip_file in download_path.glob("*.zip"):
        print(f"Extracting {zip_file.name}...")
        with zipfile.ZipFile(zip_file, 'r') as zf:
            zf.extractall(download_path)
        print(f"  Extracted successfully")

    # Check if the content is nested in a subdirectory
    subdirs = [d for d in download_path.iterdir() if d.is_dir()]
    if len(subdirs) == 1:
        nested = subdirs[0]
        nested_subdirs = [d.name.lower() for d in nested.iterdir() if d.is_dir()]
        # Check if the nested dir contains Ir/Vis
        if any(n in nested_subdirs for n in ['ir', 'vis', 'vi', 'infrared', 'visible']):
            print(f"Found dataset in nested directory: {nested.name}")
            # Nothing to do, we'll point to the nested dir


def find_m3fd_root(base_dir: str) -> str:
    """
    Find the actual M3FD root directory (containing Ir/ and Vis/ subdirs).

    Args:
        base_dir: Base directory to search in

    Returns:
        Path to M3FD root directory
    """
    base_path = Path(base_dir)

    # Check if base_dir itself has Ir/Vis
    subdirs_lower = {d.name.lower(): d for d in base_path.iterdir() if d.is_dir()}
    has_ir = any(n in subdirs_lower for n in ['ir', 'infrared', 'thermal'])
    has_vis = any(n in subdirs_lower for n in ['vis', 'vi', 'visible', 'rgb'])

    if has_ir and has_vis:
        return str(base_path)

    # Search one level deep
    for sub in base_path.iterdir():
        if sub.is_dir():
            sub_subdirs = {d.name.lower(): d for d in sub.iterdir() if d.is_dir()}
            has_ir = any(n in sub_subdirs for n in ['ir', 'infrared', 'thermal'])
            has_vis = any(n in sub_subdirs for n in ['vis', 'vi', 'visible', 'rgb'])
            if has_ir and has_vis:
                return str(sub)

    # Search two levels deep
    for sub in base_path.iterdir():
        if sub.is_dir():
            for subsub in sub.iterdir():
                if subsub.is_dir():
                    ss_subdirs = {d.name.lower(): d for d in subsub.iterdir() if d.is_dir()}
                    has_ir = any(n in ss_subdirs for n in ['ir', 'infrared', 'thermal'])
                    has_vis = any(n in ss_subdirs for n in ['vis', 'vi', 'visible', 'rgb'])
                    if has_ir and has_vis:
                        return str(subsub)

    return str(base_path)


def setup_m3fd(
    source_dir: str = None,
    download: bool = False,
    num_train: int = 200,
    num_test: int = 100,
    seed: int = 42
):
    """
    Complete M3FD setup: download (optional) + split into train/test.

    Args:
        source_dir: Path to M3FD dataset (or download target)
        download: Whether to download from Google Drive
        num_train: Number of training images
        num_test: Number of test images
        seed: Random seed for split
    """
    base_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    data_dir = base_dir / "data"
    m3fd_raw_dir = data_dir / "m3fd_raw"
    m3fd_split_dir = data_dir / "m3fd"

    # Step 1: Download if requested
    if download:
        download_m3fd(str(m3fd_raw_dir))
        source_dir = str(m3fd_raw_dir)

    if source_dir is None:
        print("ERROR: Please provide --source_dir or use --download")
        print("\nUsage:")
        print("  python setup_m3fd.py --download")
        print("  python setup_m3fd.py --source_dir /path/to/M3FD")
        sys.exit(1)

    # Step 2: Find actual M3FD root
    m3fd_root = find_m3fd_root(source_dir)
    print(f"\nM3FD root found at: {m3fd_root}")

    # Step 3: Split into train/test
    from data.m3fd_dataset import split_m3fd_dataset

    print(f"\nSplitting dataset: {num_train} train, {num_test} test")
    split_m3fd_dataset(
        source_dir=m3fd_root,
        output_dir=str(m3fd_split_dir),
        num_train=num_train,
        num_test=num_test,
        seed=seed
    )

    print("\n" + "=" * 60)
    print("M3FD SETUP COMPLETE!")
    print("=" * 60)
    print(f"  Train data: {m3fd_split_dir / 'train'}")
    print(f"  Test data:  {m3fd_split_dir / 'test'}")
    print(f"\nNext steps:")
    print(f"  1. Process training set:  python run_train.py")
    print(f"  2. Process test set:      python run_test.py")
    print(f"  3. Run everything:        python run_all.py")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Setup M3FD dataset for FusionINV"
    )
    parser.add_argument(
        "--source_dir",
        type=str,
        default=None,
        help="Path to already-downloaded M3FD dataset folder (containing Ir/ and Vis/)"
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download M3FD from Google Drive"
    )
    parser.add_argument(
        "--num_train",
        type=int,
        default=200,
        help="Number of training images (default: 200)"
    )
    parser.add_argument(
        "--num_test",
        type=int,
        default=100,
        help="Number of test images (default: 100)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for train/test split (default: 42)"
    )

    args = parser.parse_args()

    setup_m3fd(
        source_dir=args.source_dir,
        download=args.download,
        num_train=args.num_train,
        num_test=args.num_test,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
