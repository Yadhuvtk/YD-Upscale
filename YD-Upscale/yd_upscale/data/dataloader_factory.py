from __future__ import annotations

from torch.utils.data import DataLoader
from torchvision import transforms

from yd_upscale.data.dataset_paired import PairedImageDataset


def build_paired_dataloader(
    lr_manifest: str,
    hr_manifest: str,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    dataset = PairedImageDataset(
        lr_manifest=lr_manifest,
        hr_manifest=hr_manifest,
        transform=transforms.ToTensor(),
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )