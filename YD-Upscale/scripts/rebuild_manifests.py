from pathlib import Path
import random

# -------- SETTINGS --------
hr_dir = Path(r"E:\Yadhu Projects\YD-Upscale\YD-Upscale\data\rendered_hr")
lr_dir = Path(r"E:\Yadhu Projects\YD-Upscale\YD-Upscale\data\rendered_lr_x4")
manifest_dir = Path(r"E:\Yadhu Projects\YD-Upscale\YD-Upscale\data\manifests")

train_file = manifest_dir / "train.txt"
val_file = manifest_dir / "val.txt"

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
            pairs.append(rel_path.as_posix())
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

    train_file.write_text("\n".join(train_pairs) + "\n", encoding="utf-8")
    val_file.write_text("\n".join(val_pairs) + "\n", encoding="utf-8")

    print("Done.")
    print(f"Total valid pairs: {len(pairs)}")
    print(f"Train pairs: {len(train_pairs)}")
    print(f"Val pairs: {len(val_pairs)}")
    print(f"Train manifest: {train_file}")
    print(f"Val manifest: {val_file}")


if __name__ == "__main__":
    main()