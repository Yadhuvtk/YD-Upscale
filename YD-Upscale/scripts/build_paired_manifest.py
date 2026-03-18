from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build paired HR/LR train-val manifests.")
    parser.add_argument("--mapping", type=Path, required=True, help="Path to hr_lr_mapping.csv")
    parser.add_argument("--out-dir", type=Path, default=Path("data/manifests"), help="Output folder")
    parser.add_argument("--train-ratio", type=float, default=0.9, help="Train split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed")
    args = parser.parse_args()

    if not args.mapping.exists():
        raise FileNotFoundError(f"Mapping file not found: {args.mapping}")

    if not (0.0 < args.train_ratio < 1.0):
        raise ValueError("train-ratio must be between 0 and 1")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    with args.mapping.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if not rows:
        raise ValueError("No rows found in mapping file")

    rng = random.Random(args.seed)
    rng.shuffle(rows)

    train_count = int(len(rows) * args.train_ratio)
    train_rows = rows[:train_count]
    val_rows = rows[train_count:]

    files = {
        "train_hr.txt": [r["hr_path"] for r in train_rows],
        "train_lr.txt": [r["lr_path"] for r in train_rows],
        "val_hr.txt": [r["hr_path"] for r in val_rows],
        "val_lr.txt": [r["lr_path"] for r in val_rows],
    }

    for filename, lines in files.items():
        out_path = args.out_dir / filename
        with out_path.open("w", encoding="utf-8") as f:
            for line in lines:
                f.write(line + "\n")

    print("Done")
    print(f"Total pairs: {len(rows)}")
    print(f"Train pairs: {len(train_rows)}")
    print(f"Val pairs: {len(val_rows)}")
    print(f"Output dir: {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()