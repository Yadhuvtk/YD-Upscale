from __future__ import annotations

import argparse
import csv
import random
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter


VALID_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


def find_images(input_dir: Path, recursive: bool) -> list[Path]:
    if recursive:
        files = [p for p in input_dir.rglob("*") if p.is_file() and p.suffix.lower() in VALID_EXTS]
    else:
        files = [p for p in input_dir.glob("*") if p.is_file() and p.suffix.lower() in VALID_EXTS]
    return sorted(files)


def ensure_rgb(img: Image.Image) -> Image.Image:
    if img.mode != "RGB":
        return img.convert("RGB")
    return img


def apply_random_blur(img: Image.Image, rng: random.Random, blur_prob: float) -> Image.Image:
    if rng.random() < blur_prob:
        radius = rng.uniform(0.3, 1.2)
        img = img.filter(ImageFilter.GaussianBlur(radius=radius))
    return img


def apply_random_noise(img: Image.Image, rng: random.Random, noise_prob: float) -> Image.Image:
    if rng.random() >= noise_prob:
        return img

    arr = np.array(img).astype(np.float32)
    sigma = rng.uniform(1.0, 6.0)
    noise = np.random.normal(0.0, sigma, arr.shape).astype(np.float32)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def apply_random_jpeg(img: Image.Image, rng: random.Random, jpeg_prob: float) -> Image.Image:
    if rng.random() >= jpeg_prob:
        return img

    quality = rng.randint(35, 85)
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    buf.seek(0)
    out = Image.open(buf).convert("RGB")
    out.load()
    return out


def random_resize_then_downscale(
    img: Image.Image,
    scale: int,
    rng: random.Random,
    second_resize_prob: float,
) -> Image.Image:
    hr_w, hr_h = img.size
    lr_w = max(1, hr_w // scale)
    lr_h = max(1, hr_h // scale)

    resize_mode = rng.choice(
        [Image.BICUBIC, Image.BILINEAR, Image.LANCZOS]
    )

    # Main x4 downscale
    img = img.resize((lr_w, lr_h), resize_mode)

    # Optional second resize to simulate resampling artifacts
    if rng.random() < second_resize_prob:
        mid_scale = rng.uniform(0.85, 1.15)
        mid_w = max(1, int(round(lr_w * mid_scale)))
        mid_h = max(1, int(round(lr_h * mid_scale)))
        img = img.resize((mid_w, mid_h), rng.choice([Image.BICUBIC, Image.BILINEAR]))
        img = img.resize((lr_w, lr_h), rng.choice([Image.BICUBIC, Image.BILINEAR]))

    return img


def degrade_hr_to_lr(
    hr_img: Image.Image,
    scale: int,
    rng: random.Random,
    blur_prob: float,
    noise_prob: float,
    jpeg_prob: float,
    second_resize_prob: float,
) -> Image.Image:
    img = ensure_rgb(hr_img)
    img = apply_random_blur(img, rng, blur_prob)
    img = random_resize_then_downscale(img, scale, rng, second_resize_prob)
    img = apply_random_noise(img, rng, noise_prob)
    img = apply_random_jpeg(img, rng, jpeg_prob)
    return img


def save_image(img: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, format="PNG")


def process_one(
    hr_path: Path,
    lr_path: Path,
    scale: int,
    rng: random.Random,
    blur_prob: float,
    noise_prob: float,
    jpeg_prob: float,
    second_resize_prob: float,
) -> tuple[int, int, int, int]:
    with Image.open(hr_path) as hr_img:
        hr_img.load()
        hr_img = ensure_rgb(hr_img)
        hr_w, hr_h = hr_img.size

        lr_img = degrade_hr_to_lr(
            hr_img=hr_img,
            scale=scale,
            rng=rng,
            blur_prob=blur_prob,
            noise_prob=noise_prob,
            jpeg_prob=jpeg_prob,
            second_resize_prob=second_resize_prob,
        )
        lr_w, lr_h = lr_img.size
        save_image(lr_img, lr_path)

    return hr_w, hr_h, lr_w, lr_h


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create realistic degraded LR images from HR images for YD-Upscale."
    )
    parser.add_argument("--input", type=Path, required=True, help="Input HR image folder")
    parser.add_argument("--output", type=Path, required=True, help="Output LR image folder")
    parser.add_argument("--scale", type=int, default=4, help="Downscale factor. Default: 4")
    parser.add_argument("--recursive", action="store_true", help="Scan input folder recursively")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing LR images")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    parser.add_argument("--blur-prob", type=float, default=0.7, help="Probability of blur")
    parser.add_argument("--noise-prob", type=float, default=0.5, help="Probability of noise")
    parser.add_argument("--jpeg-prob", type=float, default=0.8, help="Probability of JPEG compression")
    parser.add_argument(
        "--second-resize-prob",
        type=float,
        default=0.5,
        help="Probability of extra resize artifact simulation",
    )

    parser.add_argument(
        "--mapping",
        type=Path,
        default=Path("data/manifests/hr_lr_mapping.csv"),
        help="CSV mapping output path",
    )

    args = parser.parse_args()

    input_dir = args.input.resolve()
    output_dir = args.output.resolve()
    mapping_path = args.mapping.resolve()

    if not input_dir.exists():
        raise FileNotFoundError(f"Input folder not found: {input_dir}")

    files = find_images(input_dir, args.recursive)
    if not files:
        raise FileNotFoundError(f"No images found in: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    mapping_path.parent.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)

    records = []
    created = 0
    skipped = 0
    failed = 0

    for idx, hr_path in enumerate(files, start=1):
        rel_path = hr_path.relative_to(input_dir)
        lr_path = output_dir / rel_path.with_suffix(".png")

        try:
            if lr_path.exists() and not args.overwrite:
                skipped += 1
            else:
                hr_w, hr_h, lr_w, lr_h = process_one(
                    hr_path=hr_path,
                    lr_path=lr_path,
                    scale=args.scale,
                    rng=rng,
                    blur_prob=args.blur_prob,
                    noise_prob=args.noise_prob,
                    jpeg_prob=args.jpeg_prob,
                    second_resize_prob=args.second_resize_prob,
                )
                records.append(
                    [str(hr_path), str(lr_path), hr_w, hr_h, lr_w, lr_h, args.scale]
                )
                created += 1
        except Exception as exc:
            failed += 1
            print(f"Failed: {hr_path} | {exc}")

        if idx % 100 == 0 or idx == len(files):
            print(f"[{idx}/{len(files)}] created={created} skipped={skipped} failed={failed}")

    with mapping_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["hr_path", "lr_path", "hr_width", "hr_height", "lr_width", "lr_height", "scale"]
        )
        writer.writerows(records)

    print("\nDone")
    print(f"Created LR images: {created}")
    print(f"Skipped existing: {skipped}")
    print(f"Failed: {failed}")
    print(f"Output folder: {output_dir}")
    print(f"Mapping file: {mapping_path}")


if __name__ == "__main__":
    main()