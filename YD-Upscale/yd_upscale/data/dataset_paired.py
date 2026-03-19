from __future__ import annotations

import random
from pathlib import Path
from typing import List

from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

from yd_upscale.data.degradations import degrade_for_illustration


def _read_manifest(path: Path) -> List[Path]:
    with open(path, "r", encoding="utf-8") as f:
        return [Path(line.strip()) for line in f if line.strip()]


class PairedImageDataset(Dataset):
    def __init__(
        self,
        lr_manifest: Path,
        hr_manifest: Path,
        scale: int = 4,
        patch_size_lr: int = 128,
        is_train: bool = True,
        online_degrade: bool = False,
    ):
        self.lr_paths = _read_manifest(lr_manifest)
        self.hr_paths = _read_manifest(hr_manifest)

        if len(self.lr_paths) != len(self.hr_paths):
            raise ValueError(
                f"Manifest length mismatch: LR={len(self.lr_paths)} HR={len(self.hr_paths)}"
            )

        self.scale = scale
        self.patch_size_lr = patch_size_lr
        self.is_train = is_train
        self.online_degrade = online_degrade

    def __len__(self) -> int:
        return len(self.hr_paths)

    def _load_pair(self, idx: int):
        hr = Image.open(self.hr_paths[idx]).convert("RGB")

        if self.online_degrade:
            lr = degrade_for_illustration(hr, scale=self.scale)
        else:
            lr = Image.open(self.lr_paths[idx]).convert("RGB")

        return lr, hr

    def _auto_align_pair(self, lr: Image.Image, hr: Image.Image):
        lr_w, lr_h = lr.size
        hr_w, hr_h = hr.size

        expected_hr_w = lr_w * self.scale
        expected_hr_h = lr_h * self.scale

        if hr_w == expected_hr_w and hr_h == expected_hr_h:
            return lr, hr

        # Trim to the largest valid common aligned region
        fixed_hr_w = min(hr_w, expected_hr_w)
        fixed_hr_h = min(hr_h, expected_hr_h)

        fixed_lr_w = fixed_hr_w // self.scale
        fixed_lr_h = fixed_hr_h // self.scale

        fixed_hr_w = fixed_lr_w * self.scale
        fixed_hr_h = fixed_lr_h * self.scale

        if fixed_lr_w <= 0 or fixed_lr_h <= 0:
            raise ValueError(
                f"Invalid aligned size: LR=({lr_w},{lr_h}) HR=({hr_w},{hr_h}) scale={self.scale}"
            )

        lr = lr.crop((0, 0, fixed_lr_w, fixed_lr_h))
        hr = hr.crop((0, 0, fixed_hr_w, fixed_hr_h))

        lr_w2, lr_h2 = lr.size
        hr_w2, hr_h2 = hr.size

        if hr_w2 != lr_w2 * self.scale or hr_h2 != lr_h2 * self.scale:
            raise ValueError(
                f"Scale mismatch after auto-fix: "
                f"LR=({lr_w2},{lr_h2}) HR=({hr_w2},{hr_h2}) "
                f"expected HR=({lr_w2 * self.scale},{lr_h2 * self.scale})"
            )

        return lr, hr

    def _paired_random_crop(self, lr: Image.Image, hr: Image.Image):
        lr, hr = self._auto_align_pair(lr, hr)

        lr_w, lr_h = lr.size
        hr_w, hr_h = hr.size

        expected_hr_w = lr_w * self.scale
        expected_hr_h = lr_h * self.scale

        if hr_w != expected_hr_w or hr_h != expected_hr_h:
            raise ValueError(
                f"Scale mismatch: LR=({lr_w},{lr_h}) "
                f"HR=({hr_w},{hr_h}) expected HR=({expected_hr_w},{expected_hr_h})"
            )

        lr_crop = self.patch_size_lr
        hr_crop = lr_crop * self.scale

        if lr_w < lr_crop or lr_h < lr_crop:
            raise ValueError(
                f"LR image too small for crop: LR=({lr_w},{lr_h}) patch={lr_crop}"
            )

        x_lr = random.randint(0, lr_w - lr_crop)
        y_lr = random.randint(0, lr_h - lr_crop)

        x_hr = x_lr * self.scale
        y_hr = y_lr * self.scale

        lr = lr.crop((x_lr, y_lr, x_lr + lr_crop, y_lr + lr_crop))
        hr = hr.crop((x_hr, y_hr, x_hr + hr_crop, y_hr + hr_crop))

        return lr, hr

    def _center_crop_eval(self, lr: Image.Image, hr: Image.Image):
        lr, hr = self._auto_align_pair(lr, hr)

        lr_w, lr_h = lr.size
        hr_w, hr_h = hr.size

        lr_crop = min(self.patch_size_lr, lr_w, lr_h)
        hr_crop = lr_crop * self.scale

        x_lr = max((lr_w - lr_crop) // 2, 0)
        y_lr = max((lr_h - lr_crop) // 2, 0)

        x_hr = x_lr * self.scale
        y_hr = y_lr * self.scale

        lr = lr.crop((x_lr, y_lr, x_lr + lr_crop, y_lr + lr_crop))
        hr = hr.crop((x_hr, y_hr, x_hr + hr_crop, y_hr + hr_crop))

        return lr, hr

    def _augment(self, lr: Image.Image, hr: Image.Image):
        if random.random() < 0.5:
            lr = TF.hflip(lr)
            hr = TF.hflip(hr)

        if random.random() < 0.5:
            lr = TF.vflip(lr)
            hr = TF.vflip(hr)

        rot_k = random.randint(0, 3)
        if rot_k:
            angle = 90 * rot_k
            lr = TF.rotate(lr, angle)
            hr = TF.rotate(hr, angle)

        return lr, hr

    def __getitem__(self, idx: int):
        lr, hr = self._load_pair(idx)

        if self.is_train:
            lr, hr = self._paired_random_crop(lr, hr)
            lr, hr = self._augment(lr, hr)
        else:
            lr, hr = self._center_crop_eval(lr, hr)

        lr_tensor = TF.to_tensor(lr)
        hr_tensor = TF.to_tensor(hr)

        return {
            "lr": lr_tensor,
            "hr": hr_tensor,
            "lr_path": str(self.lr_paths[idx]),
            "hr_path": str(self.hr_paths[idx]),
        }