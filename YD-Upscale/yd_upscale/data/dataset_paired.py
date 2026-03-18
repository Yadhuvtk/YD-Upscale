from __future__ import annotations

from pathlib import Path
from typing import Callable

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class PairedImageDataset(Dataset):
    def __init__(
        self,
        lr_manifest: str | Path,
        hr_manifest: str | Path,
        transform: Callable | None = None,
    ) -> None:
        self.lr_paths = self._read_manifest(lr_manifest)
        self.hr_paths = self._read_manifest(hr_manifest)

        if len(self.lr_paths) != len(self.hr_paths):
            raise ValueError(
                f"Mismatch between LR ({len(self.lr_paths)}) and HR ({len(self.hr_paths)}) samples"
            )

        self.transform = transform or transforms.ToTensor()

    def _read_manifest(self, manifest_path: str | Path) -> list[Path]:
        path = Path(manifest_path)
        if not path.exists():
            raise FileNotFoundError(f"Manifest not found: {path}")

        with path.open("r", encoding="utf-8") as f:
            lines = [Path(line.strip()) for line in f if line.strip()]

        if not lines:
            raise ValueError(f"Manifest is empty: {path}")

        return lines

    def __len__(self) -> int:
        return len(self.lr_paths)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        lr_path = self.lr_paths[index]
        hr_path = self.hr_paths[index]

        with Image.open(lr_path) as lr_img:
            lr_img = lr_img.convert("RGB")

        with Image.open(hr_path) as hr_img:
            hr_img = hr_img.convert("RGB")

        lr_tensor = self.transform(lr_img)
        hr_tensor = self.transform(hr_img)

        return {
            "lr": lr_tensor,
            "hr": hr_tensor,
            "lr_path": str(lr_path),
            "hr_path": str(hr_path),
        }