import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from yd_upscale.data.dataset_paired import PairedImageDataset

ds = PairedImageDataset(
    lr_manifest="data/manifests/train_lr.txt",
    hr_manifest="data/manifests/train_hr.txt",
)

print("Dataset length:", len(ds))
sample = ds[0]
print("LR shape:", sample["lr"].shape)
print("HR shape:", sample["hr"].shape)
print("LR path:", sample["lr_path"])
print("HR path:", sample["hr_path"])