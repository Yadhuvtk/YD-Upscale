from pathlib import Path

from yd_upscale.data.dataset_paired import PairedImageDataset


def main():
    project_root = Path(r"E:\Yadhu Projects\YD-Upscale\YD-Upscale")
    manifest_dir = project_root / "data" / "manifests"

    ds = PairedImageDataset(
        lr_list_file=manifest_dir / "train_lr.txt",
        hr_list_file=manifest_dir / "train_hr.txt",
        scale=4,
        patch_size_lr=128,
        is_train=True,
        augment=False,
    )

    sample = ds[0]

    print("Dataset length:", len(ds))
    print("LR shape:", sample["lr"].shape)
    print("HR shape:", sample["hr"].shape)
    print("LR path:", sample["lr_path"])
    print("HR path:", sample["hr_path"])


if __name__ == "__main__":
    main()