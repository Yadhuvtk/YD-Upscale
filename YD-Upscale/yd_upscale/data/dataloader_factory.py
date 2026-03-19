from __future__ import annotations

from pathlib import Path

from torch.utils.data import DataLoader

from yd_upscale.data.dataset_paired import PairedImageDataset


def build_train_val_dataloaders(
    manifest_dir: Path,
    batch_size: int = 8,
    num_workers: int = 4,
    scale: int = 4,
    patch_size_lr: int = 128,
    online_degrade: bool = False,
):
    train_dataset = PairedImageDataset(
        lr_manifest=manifest_dir / "train_lr.txt",
        hr_manifest=manifest_dir / "train_hr.txt",
        scale=scale,
        patch_size_lr=patch_size_lr,
        is_train=True,
        online_degrade=online_degrade,
    )

    val_dataset = PairedImageDataset(
        lr_manifest=manifest_dir / "val_lr.txt",
        hr_manifest=manifest_dir / "val_hr.txt",
        scale=scale,
        patch_size_lr=patch_size_lr,
        is_train=False,
        online_degrade=online_degrade,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=num_workers > 0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=num_workers > 0,
    )

    return train_loader, val_loader