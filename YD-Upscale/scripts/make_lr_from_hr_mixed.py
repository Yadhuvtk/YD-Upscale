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


def apply_gaussian_blur(img: Image.Image, rng: random.Random, sigma_range: tuple[float, float], prob: float) -> Image.Image:
    if rng.random() >= prob:
        return img
    radius = rng.uniform(*sigma_range)
    return img.filter(ImageFilter.GaussianBlur(radius=radius))


def apply_noise(img: Image.Image, rng: random.Random, sigma_range: tuple[float, float], prob: float) -> Image.Image:
    if rng.random() >= prob:
        return img

    arr = np.array(img).astype(np.float32)
    sigma = rng.uniform(*sigma_range)
    noise = np.random.normal(0.0, sigma, arr.shape).astype(np.float32)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def apply_jpeg(img: Image.Image, rng: random.Random, quality_range: tuple[int, int], prob: float) -> Image.Image:
    if rng.random() >= prob:
        return img

    q_min, q_max = quality_range
    quality = rng.randint(q_min, q_max)

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
    mid_scale_range: tuple[float, float],
) -> Image.Image:
    hr_w, hr_h = img.size
    lr_w = max(1, hr_w // scale)
    lr_h = max(1, hr_h // scale)

    main_interp = rng.choice([Image.BICUBIC, Image.BILINEAR, Image.LANCZOS])
    img = img.resize((lr_w, lr_h), main_interp)

    if rng.random() < second_resize_prob:
        mid_scale = rng.uniform(*mid_scale_range)
        mid_w = max(1, int(round(lr_w * mid_scale)))
        mid_h = max(1, int(round(lr_h * mid_scale)))

        img = img.resize((mid_w, mid_h), rng.choice([Image.BICUBIC, Image.BILINEAR]))
        img = img.resize((lr_w, lr_h), rng.choice([Image.BICUBIC, Image.BILINEAR]))

    return img


def maybe_rescale_before_downscale(
    img: Image.Image,
    rng: random.Random,
    pre_scale_prob: float,
    pre_scale_range: tuple[float, float],
) -> Image.Image:
    if rng.random() >= pre_scale_prob:
        return img

    w, h = img.size
    factor = rng.uniform(*pre_scale_range)
    new_w = max(1, int(round(w * factor)))
    new_h = max(1, int(round(h * factor)))

    interp = rng.choice([Image.BICUBIC, Image.BILINEAR, Image.LANCZOS])
    return img.resize((new_w, new_h), interp)


def get_preset_settings(preset: str) -> dict:
    if preset == "mild":
        return {
            "blur_prob": 0.45,
            "blur_sigma_range": (0.2, 0.8),
            "noise_prob": 0.20,
            "noise_sigma_range": (0.5, 2.0),
            "jpeg_prob": 0.45,
            "jpeg_quality_range": (78, 95),
            "second_resize_prob": 0.25,
            "mid_scale_range": (0.92, 1.08),
            "pre_scale_prob": 0.15,
            "pre_scale_range": (0.95, 1.05),
        }

    if preset == "medium":
        return {
            "blur_prob": 0.65,
            "blur_sigma_range": (0.4, 1.2),
            "noise_prob": 0.35,
            "noise_sigma_range": (1.0, 3.5),
            "jpeg_prob": 0.65,
            "jpeg_quality_range": (65, 88),
            "second_resize_prob": 0.45,
            "mid_scale_range": (0.88, 1.12),
            "pre_scale_prob": 0.25,
            "pre_scale_range": (0.90, 1.10),
        }

    if preset == "hard":
        return {
            "blur_prob": 0.80,
            "blur_sigma_range": (0.7, 1.8),
            "noise_prob": 0.45,
            "noise_sigma_range": (2.0, 5.0),
            "jpeg_prob": 0.80,
            "jpeg_quality_range": (50, 78),
            "second_resize_prob": 0.65,
            "mid_scale_range": (0.82, 1.18),
            "pre_scale_prob": 0.35,
            "pre_scale_range": (0.85, 1.12),
        }

    raise ValueError(f"Unknown preset: {preset}")


def choose_preset(
    rng: random.Random,
    mild_ratio: float,
    medium_ratio: float,
    hard_ratio: float,
) -> str:
    total = mild_ratio + medium_ratio + hard_ratio
    if total <= 0:
        raise ValueError("Preset ratios must sum to more than 0")

    x = rng.random() * total

    if x < mild_ratio:
        return "mild"
    if x < mild_ratio + medium_ratio:
        return "medium"
    return "hard"


def degrade_hr_to_lr(
    hr_img: Image.Image,
    scale: int,
    rng: random.Random,
    mild_ratio: float,
    medium_ratio: float,
    hard_ratio: float,
) -> tuple[Image.Image, str]:
    img = ensure_rgb(hr_img)

    preset = choose_preset(rng, mild_ratio, medium_ratio, hard_ratio)
    cfg = get_preset_settings(preset)

    img = maybe_rescale_before_downscale(
        img,
        rng=rng,
        pre_scale_prob=cfg["pre_scale_prob"],
        pre_scale_range=cfg["pre_scale_range"],
    )

    img = apply_gaussian_blur(
        img,
        rng=rng,
        sigma_range=cfg["blur_sigma_range"],
        prob=cfg["blur_prob"],
    )

    img = random_resize_then_downscale(
        img,
        scale=scale,
        rng=rng,
        second_resize_prob=cfg["second_resize_prob"],
        mid_scale_range=cfg["mid_scale_range"],
    )

    img = apply_noise(
        img,
        rng=rng,
        sigma_range=cfg["noise_sigma_range"],
        prob=cfg["noise_prob"],
    )

    img = apply_jpeg(
        img,
        rng=rng,
        quality_range=cfg["jpeg_quality_range"],
        prob=cfg["jpeg_prob"],
    )

    return img, preset


def save_image(img: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, format="PNG")


def process_one(
    hr_path: Path,
    lr_path: Path,
    scale: int,
    rng: random.Random,
    mild_ratio: float,
    medium_ratio: float,
    hard_ratio: float,
) -> tuple[int, int, int, int, str]:
    with Image.open(hr_path) as hr_img:
        hr_img.load()
        hr_img = ensure_rgb(hr_img)
        hr_w, hr_h = hr_img.size

        lr_img, preset = degrade_hr_to_lr(
            hr_img=hr_img,
            scale=scale,
            rng=rng,
            mild_ratio=mild_ratio,
            medium_ratio=medium_ratio,
            hard_ratio=hard_ratio,
        )

        lr_w, lr_h = lr_img.size
        save_image(lr_img, lr_path)

    return hr_w, hr_h, lr_w, lr_h, preset


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create mixed realistic LR images from HR images for YD-Upscale."
    )
    parser.add_argument("--input", type=Path, required=True, help="Input HR image folder")
    parser.add_argument("--output", type=Path, required=True, help="Output LR image folder")
    parser.add_argument("--scale", type=int, default=4, help="Downscale factor. Default: 4")
    parser.add_argument("--recursive", action="store_true", help="Scan input folder recursively")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing LR images")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    parser.add_argument("--mild-ratio", type=float, default=0.50, help="Ratio for mild preset")
    parser.add_argument("--medium-ratio", type=float, default=0.35, help="Ratio for medium preset")
    parser.add_argument("--hard-ratio", type=float, default=0.15, help="Ratio for hard preset")

    parser.add_argument(
        "--mapping",
        type=Path,
        default=Path("data/manifests/hr_lr_mapping_mixed.csv"),
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
    preset_counts = {"mild": 0, "medium": 0, "hard": 0}

    for idx, hr_path in enumerate(files, start=1):
        rel_path = hr_path.relative_to(input_dir)
        lr_path = output_dir / rel_path.with_suffix(".png")

        try:
            if lr_path.exists() and not args.overwrite:
                skipped += 1
            else:
                hr_w, hr_h, lr_w, lr_h, preset = process_one(
                    hr_path=hr_path,
                    lr_path=lr_path,
                    scale=args.scale,
                    rng=rng,
                    mild_ratio=args.mild_ratio,
                    medium_ratio=args.medium_ratio,
                    hard_ratio=args.hard_ratio,
                )
                preset_counts[preset] += 1
                records.append(
                    [
                        str(hr_path),
                        str(lr_path),
                        hr_w,
                        hr_h,
                        lr_w,
                        lr_h,
                        args.scale,
                        preset,
                    ]
                )
                created += 1
        except Exception as exc:
            failed += 1
            print(f"Failed: {hr_path} | {exc}")

        if idx % 100 == 0 or idx == len(files):
            print(
                f"[{idx}/{len(files)}] created={created} skipped={skipped} failed={failed} "
                f"mild={preset_counts['mild']} medium={preset_counts['medium']} hard={preset_counts['hard']}"
            )

    with mapping_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "hr_path",
                "lr_path",
                "hr_width",
                "hr_height",
                "lr_width",
                "lr_height",
                "scale",
                "preset",
            ]
        )
        writer.writerows(records)

    print("\nDone")
    print(f"Created LR images: {created}")
    print(f"Skipped existing: {skipped}")
    print(f"Failed: {failed}")
    print(f"Preset counts: {preset_counts}")
    print(f"Output folder: {output_dir}")
    print(f"Mapping file: {mapping_path}")


if __name__ == "__main__":
    main()