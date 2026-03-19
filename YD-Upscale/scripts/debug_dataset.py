"""
Dataset Debug Script — YD-Upscale

Randomly samples N LR/HR pairs from your manifests and:
  - Saves side-by-side comparison: [LR upscaled | HR | difference heatmap]
  - Prints size info and pixel difference statistics

KEY METRIC to watch:
  MAE(bicubic_upsample(LR), HR)
  - If MAE < 5  → pairs are TOO SIMILAR, model has nothing to learn. Re-generate LR.
  - If MAE 5-20 → degradation is mild but usable
  - If MAE > 20 → good degradation level for meaningful training

Usage:
    python scripts/debug_dataset.py
    python scripts/debug_dataset.py --n 20 --manifest-dir data/manifests --output debug_output
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def read_manifest(path: Path) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = np.mean((a.astype(float) - b.astype(float)) ** 2)
    if mse == 0:
        return float("inf")
    return 20.0 * np.log10(255.0 / np.sqrt(mse))


def make_comparison_strip(lr_img: Image.Image, hr_img: Image.Image, scale: int) -> Image.Image:
    """
    Returns a side-by-side strip:
      [bicubic upsample of LR | HR ground truth | absolute difference x5]
    """
    # Upsample LR to HR size using bicubic (this is what a naive resize would give)
    lr_up = lr_img.resize(hr_img.size, Image.BICUBIC)

    lr_arr = np.array(lr_up).astype(np.int16)
    hr_arr = np.array(hr_img).astype(np.int16)

    diff = np.abs(lr_arr - hr_arr).astype(np.uint8)
    # Amplify difference 5x so subtle differences become visible
    diff_amplified = np.clip(diff * 5, 0, 255).astype(np.uint8)
    diff_img = Image.fromarray(diff_amplified)

    w, h = hr_img.size
    strip = Image.new("RGB", (w * 3, h))
    strip.paste(lr_up,    (0,     0))
    strip.paste(hr_img,   (w,     0))
    strip.paste(diff_img, (w * 2, 0))

    return strip


def add_label(img: Image.Image, labels: list[str]) -> Image.Image:
    """Add text labels at top of each panel."""
    try:
        import cv2 as _cv2
        arr = np.array(img)
        panel_w = arr.shape[1] // len(labels)
        for i, label in enumerate(labels):
            x = i * panel_w + 10
            _cv2.putText(arr, label, (x, 30), _cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
        return Image.fromarray(arr)
    except Exception:
        return img  # skip labels if cv2 isn't available


def main():
    parser = argparse.ArgumentParser(description="Debug LR/HR dataset quality")
    parser.add_argument("--n", type=int, default=10, help="Number of pairs to inspect")
    parser.add_argument("--manifest-dir", type=Path, default=Path("data/manifests"))
    parser.add_argument("--output", type=Path, default=Path("debug_output"))
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    manifest_dir = args.manifest_dir
    lr_manifest = manifest_dir / "train_lr.txt"
    hr_manifest = manifest_dir / "train_hr.txt"

    if not lr_manifest.exists() or not hr_manifest.exists():
        print(f"ERROR: manifests not found in {manifest_dir}")
        print(f"  Expected: {lr_manifest}")
        print(f"  Expected: {hr_manifest}")
        sys.exit(1)

    lr_paths = read_manifest(lr_manifest)
    hr_paths = read_manifest(hr_manifest)

    if len(lr_paths) != len(hr_paths):
        print(f"ERROR: manifest size mismatch: LR={len(lr_paths)} HR={len(hr_paths)}")
        sys.exit(1)

    n = min(args.n, len(lr_paths))
    rng = random.Random(args.seed)
    indices = rng.sample(range(len(lr_paths)), n)

    args.output.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  YD-Upscale Dataset Debug  |  {n} random pairs")
    print(f"  MAE key: <5=too_similar  5-20=mild  >20=good_degradation")
    print(f"{'='*70}")
    print(f"{'#':<4} {'MAE':>7} {'PSNR(dB)':>10} {'LR size':>12} {'HR size':>12}  LR file")
    print(f"{'-'*70}")

    maes = []
    psnrs = []

    for i, idx in enumerate(indices):
        lr_path = Path(lr_paths[idx])
        hr_path = Path(hr_paths[idx])

        try:
            lr_img = Image.open(lr_path).convert("RGB")
            hr_img = Image.open(hr_path).convert("RGB")
        except Exception as e:
            print(f"{i+1:<4} LOAD ERROR: {e}")
            continue

        lr_up = lr_img.resize(hr_img.size, Image.BICUBIC)
        lr_arr = np.array(lr_up).astype(np.int16)
        hr_arr = np.array(hr_img).astype(np.int16)

        mae = float(np.mean(np.abs(lr_arr - hr_arr)))
        p = psnr(np.array(lr_up), np.array(hr_img))

        maes.append(mae)
        psnrs.append(p)

        flag = ""
        if mae < 5:
            flag = "  ← TOO SIMILAR"
        elif mae > 20:
            flag = "  ✓ good"

        print(
            f"{i+1:<4} {mae:>7.2f} {p:>10.2f}"
            f" {str(lr_img.size):>12} {str(hr_img.size):>12}"
            f"  {lr_path.name}{flag}"
        )

        # Save side-by-side strip
        strip = make_comparison_strip(lr_img, hr_img, scale=args.scale)
        strip = add_label(strip, ["bicubic(LR)", "HR ground truth", "diff x5"])
        out_path = args.output / f"pair_{i+1:02d}_mae{mae:.1f}.png"
        strip.save(out_path)

    print(f"\n{'='*70}")
    print(f"  SUMMARY over {len(maes)} pairs:")
    if maes:
        print(f"  MAE   — mean: {np.mean(maes):.2f}  min: {np.min(maes):.2f}  max: {np.max(maes):.2f}")
        print(f"  PSNR  — mean: {np.mean(psnrs):.2f} dB")
        if np.mean(maes) < 5:
            print()
            print("  !! DIAGNOSIS: LR-HR pairs are nearly identical.")
            print("     The model has almost nothing to learn.")
            print("     ACTION: Re-generate LR with stronger degradation, OR")
            print("             use --online-degrade flag in train.py")
        elif np.mean(maes) < 10:
            print()
            print("  ~ Degradation is mild. Model can learn but slowly.")
            print("    Consider: increase JPEG strength or use --online-degrade")
        else:
            print()
            print("  ✓ Degradation level looks meaningful. Proceed with training.")
    print(f"\n  Comparison strips saved to: {args.output.resolve()}")
    print(f"  Each strip: [bicubic(LR) | HR | diff×5]")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
