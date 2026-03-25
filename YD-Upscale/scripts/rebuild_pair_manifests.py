from pathlib import Path
import random

# -------- SETTINGS --------
hr_dir = Path(r"E:\Yadhu Projects\YD-Upscale\YD-Upscale\data\rendered_hr")
lr_dir = Path(r"E:\Yadhu Projects\YD-Upscale\YD-Upscale\data\rendered_lr_x4")
manifest_dir = Path(r"E:\Yadhu Projects\YD-Upscale\YD-Upscale\data\manifests")

train_hr_file = manifest_dir / "train_hr.txt"
train_lr_file = manifest_dir / "train_lr.txt"
val_hr_file = manifest_dir / "val_hr.txt"
val_lr_file = manifest_dir / "val_lr.txt"

val_ratio = 0.10
seed = 42
# --------------------------


def main():
    manifest_dir.mkdir(parents=True, exist_ok=True)

    hr_files = sorted(hr_dir.rglob("*.png"))
    if not hr_files:
        print(f"No HR files found in: {hr_dir}")
        return

    pairs = []

    for hr_path in hr_files:
        rel_path = hr_path.relative_to(hr_dir)
        lr_path = lr_dir / rel_path

        if lr_path.exists():
            pairs.append((hr_path.resolve(), lr_path.resolve()))
        else:
            print(f"Missing LR for: {rel_path.as_posix()}")

    if not pairs:
        print("No valid HR/LR pairs found.")
        return

    random.seed(seed)
    random.shuffle(pairs)

    val_count = int(len(pairs) * val_ratio)
    val_pairs = pairs[:val_count]
    train_pairs = pairs[val_count:]

    train_hr_lines = [str(hr).replace("\\", "/") for hr, lr in train_pairs]
    train_lr_lines = [str(lr).replace("\\", "/") for hr, lr in train_pairs]
    val_hr_lines = [str(hr).replace("\\", "/") for hr, lr in val_pairs]
    val_lr_lines = [str(lr).replace("\\", "/") for hr, lr in val_pairs]

    train_hr_file.write_text("\n".join(train_hr_lines) + "\n", encoding="utf-8")
    train_lr_file.write_text("\n".join(train_lr_lines) + "\n", encoding="utf-8")
    val_hr_file.write_text("\n".join(val_hr_lines) + "\n", encoding="utf-8")
    val_lr_file.write_text("\n".join(val_lr_lines) + "\n", encoding="utf-8")

    print("Done.")
    print(f"Total valid pairs: {len(pairs)}")
    print(f"Train pairs: {len(train_pairs)}")
    print(f"Val pairs: {len(val_pairs)}")
    print(f"train_hr: {train_hr_file}")
    print(f"train_lr: {train_lr_file}")
    print(f"val_hr:   {val_hr_file}")
    print(f"val_lr:   {val_lr_file}")


if __name__ == "__main__":
    main()