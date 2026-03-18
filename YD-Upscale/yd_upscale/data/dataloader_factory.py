from __future__ import annotations

from pathlib import Path
from torch.utils.data import DataLoader

from yd_upscale.data.dataset_paired import PairedImageDataset


def build_train_val_dataloaders(
    manifest_dir: str | Path,
    batch_size: int = 8,
    num_workers: int = 4,
    scale: int = 4,
    patch_size_lr: int = 128,
):
    manifest_dir = Path(manifest_dir)

    train_dataset = PairedImageDataset(
        lr_list_file=manifest_dir / "train_lr.txt",
        hr_list_file=manifest_dir / "train_hr.txt",
        scale=scale,
        patch_size_lr=patch_size_lr,
        is_train=True,
        augment=True,
    )

    val_dataset = PairedImageDataset(
        lr_list_file=manifest_dir / "val_lr.txt",
        hr_list_file=manifest_dir / "val_hr.txt",
        scale=scale,
        patch_size_lr=patch_size_lr,
        is_train=False,
        augment=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader