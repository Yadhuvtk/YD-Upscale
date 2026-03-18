from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Rename PNG images sequentially.")
    parser.add_argument("--input", type=Path, required=True, help="Folder containing PNG files")
    parser.add_argument("--start", type=int, default=1, help="Starting number")
    parser.add_argument("--prefix", type=str, default="", help="Optional filename prefix")
    parser.add_argument(
        "--mapping",
        type=Path,
        default=Path("data/manifests/rename_mapping.csv"),
        help="CSV file to save old->new mapping",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Rename PNGs recursively",
    )
    args = parser.parse_args()

    input_dir = args.input.resolve()
    mapping_path = args.mapping.resolve()
    mapping_path.parent.mkdir(parents=True, exist_ok=True)

    if args.recursive:
        files = sorted(input_dir.rglob("*.png"))
    else:
        files = sorted(input_dir.glob("*.png"))

    if not files:
        raise FileNotFoundError(f"No PNG files found in: {input_dir}")

    temp_records = []
    number = args.start

    # First rename to temporary names to avoid collisions
    for file_path in files:
        temp_path = file_path.with_name(f"__tmp__{number:06d}.png")
        file_path.rename(temp_path)
        temp_records.append((file_path, temp_path, number))
        number += 1

    # Then rename to final serial names
    final_records = []
    for original_path, temp_path, idx in temp_records:
        new_name = f"{args.prefix}{idx}.png" if args.prefix else f"{idx}.png"
        final_path = temp_path.with_name(new_name)
        temp_path.rename(final_path)
        final_records.append((str(original_path), str(final_path), idx))

    with mapping_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["old_path", "new_path", "serial_number"])
        writer.writerows(final_records)

    print(f"Renamed {len(final_records)} files.")
    print(f"Mapping saved to: {mapping_path}")


if __name__ == "__main__":
    main()