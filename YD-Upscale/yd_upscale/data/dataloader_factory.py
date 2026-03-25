"""
dataloader_factory.py — DataLoader builder for anime super-resolution.

Data pipeline
-------------
Reads HR image paths from manifest text files (one path per line), then:

  Training:
    1. Load HR image (RGB)
    2. Random crop to 256×256 (HR patch size)
    3. Random horizontal flip (p=0.5)
    4. Random 90° rotation (0°, 90°, 180°, or 270°)
    5. Bicubic downsample ×4 → 64×64 LR patch

  Validation:
    1. Load HR image (RGB)
    2. Centre crop to 256×256
    3. Bicubic downsample ×4 → 64×64 LR patch
    (No augmentation)

LR is always generated on-the-fly from the HR patch, guaranteeing
perfectly aligned pairs and clean bicubic degradation.

Manifest format
---------------
Plain .txt file, one image path per line:
    /data/anime_hr/img001.png
    /data/anime_hr/img002.png

Expected files in manifest_dir:
    train_hr.txt, val_hr.txt
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List

import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import DataLoader, Dataset


# ---------------------------------------------------------------------------
# Manifest reader
# ---------------------------------------------------------------------------

def _load_manifest(path: Path) -> List[Path]:
    """Read manifest file → list of image paths (blank lines skipped)."""
    with open(path, "r", encoding="utf-8") as f:
        return [Path(line.strip()) for line in f if line.strip()]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class AnimeHRDataset(Dataset):
    """
    HR-only dataset. LR is generated on-the-fly via bicubic downsampling.

    Args:
        manifest_path : .txt file listing HR image paths
        hr_crop_size  : square crop size for HR patches (default 256)
        scale         : LR downscale factor (default 4)
        is_train      : True → random crop + augment, False → centre crop
    """

    def __init__(
        self,
        manifest_path: Path,
        hr_crop_size: int = 256,
        scale: int = 4,
        is_train: bool = True,
    ):
        self.hr_crop   = hr_crop_size
        self.lr_crop   = hr_crop_size // scale    # 256 / 4 = 64
        self.scale     = scale
        self.is_train  = is_train

        # Load and filter — skip images too small for a full crop
        raw_paths = _load_manifest(manifest_path)
        valid: List[Path] = []
        skipped = 0
        for p in raw_paths:
            try:
                with Image.open(p) as img:
                    w, h = img.size
                if w >= hr_crop_size and h >= hr_crop_size:
                    valid.append(p)
                else:
                    skipped += 1
            except Exception:
                skipped += 1

        self.hr_paths = valid
        split = "train" if is_train else "val"
        print(
            f"[AnimeHRDataset][{split}] {len(self.hr_paths)} images "
            f"(skipped {skipped} too-small/unreadable)"
        )

    def __len__(self) -> int:
        return len(self.hr_paths)

    # ----- Crop helpers -----

    def _random_crop(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        x = random.randint(0, w - self.hr_crop)
        y = random.randint(0, h - self.hr_crop)
        return img.crop((x, y, x + self.hr_crop, y + self.hr_crop))

    def _centre_crop(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        x = (w - self.hr_crop) // 2
        y = (h - self.hr_crop) // 2
        return img.crop((x, y, x + self.hr_crop, y + self.hr_crop))

    # ----- Augmentation -----

    def _augment(self, img: Image.Image) -> Image.Image:
        """Random horizontal flip + random 90° rotation."""
        if random.random() < 0.5:
            img = TF.hflip(img)
        k = random.randint(0, 3)
        if k:
            img = TF.rotate(img, angle=90 * k)
        return img

    # ----- LR generation -----

    def _make_lr(self, hr: Image.Image) -> Image.Image:
        """Bicubic downsample HR → LR."""
        return hr.resize((self.lr_crop, self.lr_crop), Image.BICUBIC)

    # ----- __getitem__ -----

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        hr_img = Image.open(self.hr_paths[idx]).convert("RGB")

        # Crop
        if self.is_train:
            hr_patch = self._random_crop(hr_img)
            hr_patch = self._augment(hr_patch)
        else:
            hr_patch = self._centre_crop(hr_img)

        # Generate LR from HR (always after augmentation so they match)
        lr_patch = self._make_lr(hr_patch)

        return {
            "lr": TF.to_tensor(lr_patch),    # (3, 64, 64)   in [0,1]
            "hr": TF.to_tensor(hr_patch),    # (3, 256, 256) in [0,1]
            "hr_path": str(self.hr_paths[idx]),
        }


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_train_val_dataloaders(
    manifest_dir: Path,
    batch_size: int = 16,
    num_workers: int = 4,
    scale: int = 4,
    hr_crop_size: int = 256,
) -> tuple[DataLoader, DataLoader]:
    """
    Build train and val DataLoaders.

    Args:
        manifest_dir  : folder containing train_hr.txt and val_hr.txt
        batch_size    : training batch size (val fixed at 1)
        num_workers   : DataLoader workers
        scale         : SR scale factor
        hr_crop_size  : HR patch crop size

    Returns:
        (train_loader, val_loader)
    """
    train_ds = AnimeHRDataset(
        manifest_dir / "train_hr.txt", hr_crop_size, scale, is_train=True,
    )
    val_ds = AnimeHRDataset(
        manifest_dir / "val_hr.txt", hr_crop_size, scale, is_train=False,
    )

    common = dict(
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=True, **common,
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False, drop_last=False, **common,
    )
    return train_loader, val_loader
