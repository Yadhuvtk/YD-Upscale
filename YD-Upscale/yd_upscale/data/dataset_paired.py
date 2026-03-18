from __future__ import annotations

import random
from pathlib import Path
from typing import List, Tuple, Optional

from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF


def read_path_list(txt_file: str | Path) -> List[str]:
    txt_file = Path(txt_file)
    with txt_file.open("r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    return lines


class PairedImageDataset(Dataset):
    """
    Paired LR/HR dataset for x4 super-resolution.

    Training:
        - random paired crop
        - optional random hflip / vflip / 90deg rotation

    Validation:
        - center crop
        OR
        - full image if patch_size_lr is None
    """

    def __init__(
        self,
        lr_list_file: str | Path,
        hr_list_file: str | Path,
        scale: int = 4,
        patch_size_lr: Optional[int] = 128,
        is_train: bool = True,
        augment: bool = True,
    ) -> None:
        self.lr_paths = read_path_list(lr_list_file)
        self.hr_paths = read_path_list(hr_list_file)

        if len(self.lr_paths) != len(self.hr_paths):
            raise ValueError(
                f"LR/HR list size mismatch: {len(self.lr_paths)} vs {len(self.hr_paths)}"
            )

        self.scale = scale
        self.patch_size_lr = patch_size_lr
        self.patch_size_hr = patch_size_lr * scale if patch_size_lr is not None else None
        self.is_train = is_train
        self.augment = augment and is_train

    def __len__(self) -> int:
        return len(self.lr_paths)

    def _align_pair(self, lr: Image.Image, hr: Image.Image) -> Tuple[Image.Image, Image.Image]:
        """
        Safely align LR/HR pair to valid x4 relationship by trimming excess pixels.
        This handles rare off-by-1 or rounding mismatch cases.
        """
        lr_w, lr_h = lr.size
        hr_w, hr_h = hr.size

        valid_lr_w = min(lr_w, hr_w // self.scale)
        valid_lr_h = min(lr_h, hr_h // self.scale)

        if valid_lr_w <= 0 or valid_lr_h <= 0:
            raise ValueError(
                f"Invalid aligned size: LR=({lr_w},{lr_h}) HR=({hr_w},{hr_h}) scale={self.scale}"
            )

        valid_hr_w = valid_lr_w * self.scale
        valid_hr_h = valid_lr_h * self.scale

        if (lr_w, lr_h) != (valid_lr_w, valid_lr_h):
            lr = lr.crop((0, 0, valid_lr_w, valid_lr_h))

        if (hr_w, hr_h) != (valid_hr_w, valid_hr_h):
            hr = hr.crop((0, 0, valid_hr_w, valid_hr_h))

        return lr, hr

    def _load_pair(self, index: int) -> Tuple[Image.Image, Image.Image]:
        lr_path = self.lr_paths[index]
        hr_path = self.hr_paths[index]

        lr = Image.open(lr_path).convert("RGB")
        hr = Image.open(hr_path).convert("RGB")

        lr, hr = self._align_pair(lr, hr)

        return lr, hr

    def _paired_random_crop(
        self, lr: Image.Image, hr: Image.Image
    ) -> Tuple[Image.Image, Image.Image]:
        if self.patch_size_lr is None:
            return lr, hr

        lr_w, lr_h = lr.size
        hr_w, hr_h = hr.size

        expected_hr_w = lr_w * self.scale
        expected_hr_h = lr_h * self.scale

        if hr_w != expected_hr_w or hr_h != expected_hr_h:
            raise ValueError(
                f"Scale mismatch: LR=({lr_w},{lr_h}) HR=({hr_w},{hr_h}) "
                f"expected HR=({expected_hr_w},{expected_hr_h})"
            )

        if lr_w < self.patch_size_lr or lr_h < self.patch_size_lr:
            raise ValueError(
                f"LR image too small for crop: image=({lr_w},{lr_h}) "
                f"patch={self.patch_size_lr}"
            )

        x_lr = random.randint(0, lr_w - self.patch_size_lr)
        y_lr = random.randint(0, lr_h - self.patch_size_lr)

        x_hr = x_lr * self.scale
        y_hr = y_lr * self.scale

        lr_crop = lr.crop(
            (x_lr, y_lr, x_lr + self.patch_size_lr, y_lr + self.patch_size_lr)
        )
        hr_crop = hr.crop(
            (x_hr, y_hr, x_hr + self.patch_size_hr, y_hr + self.patch_size_hr)
        )

        return lr_crop, hr_crop

    def _paired_center_crop(
        self, lr: Image.Image, hr: Image.Image
    ) -> Tuple[Image.Image, Image.Image]:
        if self.patch_size_lr is None:
            return lr, hr

        lr_w, lr_h = lr.size
        hr_w, hr_h = hr.size

        expected_hr_w = lr_w * self.scale
        expected_hr_h = lr_h * self.scale

        if hr_w != expected_hr_w or hr_h != expected_hr_h:
            raise ValueError(
                f"Scale mismatch: LR=({lr_w},{lr_h}) HR=({hr_w},{hr_h}) "
                f"expected HR=({expected_hr_w},{expected_hr_h})"
            )

        if lr_w < self.patch_size_lr or lr_h < self.patch_size_lr:
            raise ValueError(
                f"LR image too small for crop: image=({lr_w},{lr_h}) "
                f"patch={self.patch_size_lr}"
            )

        x_lr = (lr_w - self.patch_size_lr) // 2
        y_lr = (lr_h - self.patch_size_lr) // 2

        x_hr = x_lr * self.scale
        y_hr = y_lr * self.scale

        lr_crop = lr.crop(
            (x_lr, y_lr, x_lr + self.patch_size_lr, y_lr + self.patch_size_lr)
        )
        hr_crop = hr.crop(
            (x_hr, y_hr, x_hr + self.patch_size_hr, y_hr + self.patch_size_hr)
        )

        return lr_crop, hr_crop

    def _augment_pair(
        self, lr: Image.Image, hr: Image.Image
    ) -> Tuple[Image.Image, Image.Image]:
        if not self.augment:
            return lr, hr

        if random.random() < 0.5:
            lr = TF.hflip(lr)
            hr = TF.hflip(hr)

        if random.random() < 0.5:
            lr = TF.vflip(lr)
            hr = TF.vflip(hr)

        if random.random() < 0.5:
            lr = lr.transpose(Image.Transpose.ROTATE_90)
            hr = hr.transpose(Image.Transpose.ROTATE_90)

        return lr, hr

    def __getitem__(self, index: int):
        lr, hr = self._load_pair(index)

        if self.is_train:
            lr, hr = self._paired_random_crop(lr, hr)
            lr, hr = self._augment_pair(lr, hr)
        else:
            lr, hr = self._paired_center_crop(lr, hr)

        lr_tensor = TF.to_tensor(lr)
        hr_tensor = TF.to_tensor(hr)

        return {
            "lr": lr_tensor,
            "hr": hr_tensor,
            "lr_path": self.lr_paths[index],
            "hr_path": self.hr_paths[index],
        }