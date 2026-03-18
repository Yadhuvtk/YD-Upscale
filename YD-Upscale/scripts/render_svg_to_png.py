from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
from pathlib import Path
from typing import Iterable

from PIL import Image

DEFAULT_INPUT = Path(r"E:\Yadhu Projects\New-Svgs")
DEFAULT_OUTPUT = Path("data/rendered_hr")
DEFAULT_MANIFEST = Path("data/manifests/render_manifest.jsonl")
DEFAULT_FAILURES = Path("data/manifests/render_failures.jsonl")


def setup_logger(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def find_svg_files(input_dir: Path) -> list[Path]:
    return sorted(p for p in input_dir.rglob("*.svg") if p.is_file())


def verify_png(image_path: Path) -> tuple[int, int]:
    with Image.open(image_path) as img:
        img.load()
        return img.width, img.height


def resolve_inkscape_path(inkscape_path: str) -> str:
    if inkscape_path.lower() == "inkscape":
        resolved = shutil.which("inkscape")
        if resolved is None:
            raise FileNotFoundError(
                "Inkscape executable not found. Pass --inkscape-path or add Inkscape to PATH."
            )
        return resolved

    candidate = Path(inkscape_path)
    if not candidate.exists():
        raise FileNotFoundError(
            f"Inkscape executable not found at: {candidate}. "
            "Pass a valid --inkscape-path."
        )
    return str(candidate)


def render_with_inkscape(
    svg_path: Path,
    png_path: Path,
    size: int,
    inkscape_path: str,
    background: str = "white",
    fit_mode: str = "pad",
) -> None:
    png_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        inkscape_path,
        str(svg_path),
        "--export-type=png",
        f"--export-filename={png_path}",
    ]

    if fit_mode in {"width", "pad"}:
        cmd.append(f"--export-width={size}")
    elif fit_mode == "height":
        cmd.append(f"--export-height={size}")
    else:
        raise ValueError(f"Unsupported fit_mode: {fit_mode}")

    if background == "white":
        cmd.extend(["--export-background=white", "--export-background-opacity=1.0"])
    elif background == "transparent":
        cmd.extend(["--export-background-opacity=0.0"])
    else:
        raise ValueError(f"Unsupported background: {background}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
            timeout=60,
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"Inkscape render timed out after 60 seconds: {svg_path}")

    if result.returncode != 0:
        stderr = result.stderr.strip() or result.stdout.strip() or "Unknown Inkscape error"
        raise RuntimeError(stderr)


def pad_to_square(image_path: Path, size: int, background: str) -> tuple[int, int]:
    with Image.open(image_path) as img:
        img.load()

        if background == "transparent":
            if img.mode != "RGBA":
                img = img.convert("RGBA")
            canvas = Image.new("RGBA", (size, size), (255, 255, 255, 0))
        else:
            if img.mode != "RGB":
                img = img.convert("RGB")
            canvas = Image.new("RGB", (size, size), (255, 255, 255))

        src_w, src_h = img.size

        if src_w > size or src_h > size:
            scale = min(size / src_w, size / src_h)
            new_w = max(1, int(round(src_w * scale)))
            new_h = max(1, int(round(src_h * scale)))
            img = img.resize((new_w, new_h), Image.LANCZOS)
            src_w, src_h = img.size

        offset_x = (size - src_w) // 2
        offset_y = (size - src_h) // 2

        if img.mode == "RGBA":
            canvas.paste(img, (offset_x, offset_y), img)
        else:
            canvas.paste(img, (offset_x, offset_y))

        canvas.save(image_path)

    return verify_png(image_path)


def write_jsonl_record(file_obj, record: dict) -> None:
    file_obj.write(json.dumps(record, ensure_ascii=False) + "\n")


def iter_svg_files(files: list[Path], limit: int | None) -> Iterable[Path]:
    if limit is None:
        yield from files
    else:
        yield from files[:limit]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Render SVG files into PNG images for YD-Upscale dataset preparation.\n\n"
            "Examples:\n"
            r'  python scripts\render_svg_to_png.py --limit 10 --size 2048 '
            r'--inkscape-path "C:\Program Files\Inkscape\bin\inkscape.exe"' "\n"
            r'  python scripts\render_svg_to_png.py --size 2048 --fit-mode pad '
            r'--background white --inkscape-path "C:\Program Files\Inkscape\bin\inkscape.exe"'
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Input SVG root folder. Default: {DEFAULT_INPUT}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output PNG root folder. Default: {DEFAULT_OUTPUT}",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help=f"Render manifest JSONL path. Default: {DEFAULT_MANIFEST}",
    )
    parser.add_argument(
        "--failures",
        type=Path,
        default=DEFAULT_FAILURES,
        help=f"Failures JSONL path. Default: {DEFAULT_FAILURES}",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=2048,
        help="Target render size. Default: 2048",
    )
    parser.add_argument(
        "--inkscape-path",
        type=str,
        default="inkscape",
        help='Inkscape executable path. Example: "C:\\Program Files\\Inkscape\\bin\\inkscape.exe"',
    )
    parser.add_argument(
        "--background",
        choices=["white", "transparent"],
        default="white",
        help="PNG background mode. Default: white",
    )
    parser.add_argument(
        "--fit-mode",
        choices=["width", "height", "pad"],
        default="pad",
        help=(
            "How to preserve aspect ratio.\n"
            "width: render with only export-width\n"
            "height: render with only export-height\n"
            "pad: render proportionally using width, then pad to square"
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing PNG files.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N SVG files for testing.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )

    args = parser.parse_args()
    setup_logger(args.verbose)

    input_dir = args.input.resolve()
    output_dir = args.output.resolve()
    manifest_path = args.manifest.resolve()
    failures_path = args.failures.resolve()

    if not input_dir.exists():
        raise FileNotFoundError(f"Input folder not found: {input_dir}")

    resolved_inkscape = resolve_inkscape_path(args.inkscape_path)

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    failures_path.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    svg_files = find_svg_files(input_dir)
    if not svg_files:
        raise FileNotFoundError(f"No SVG files found in: {input_dir}")

    total_target = min(len(svg_files), args.limit) if args.limit else len(svg_files)
    logging.info("Found %s SVG files. Processing %s.", len(svg_files), total_target)

    success_count = 0
    fail_count = 0
    skipped_count = 0

    with manifest_path.open("w", encoding="utf-8") as manifest_f, failures_path.open(
        "w", encoding="utf-8"
    ) as failures_f:
        for index, svg_path in enumerate(iter_svg_files(svg_files, args.limit), start=1):
            rel_path = svg_path.relative_to(input_dir)
            png_path = output_dir / rel_path.with_suffix(".png")

            logging.info("Rendering [%s/%s]: %s", index, total_target, svg_path)

            try:
                if png_path.exists() and not args.overwrite:
                    try:
                        width, height = verify_png(png_path)
                        skipped_count += 1
                        logging.info(
                            "Skipped existing [%s/%s]: %s (%sx%s)",
                            index,
                            total_target,
                            png_path,
                            width,
                            height,
                        )
                        write_jsonl_record(
                            manifest_f,
                            {
                                "source_svg": str(svg_path),
                                "rendered_png": str(png_path),
                                "width": width,
                                "height": height,
                                "status": "skipped_existing",
                            },
                        )
                    except Exception:
                        logging.warning("Existing PNG is corrupted, re-rendering: %s", png_path)
                        render_with_inkscape(
                            svg_path=svg_path,
                            png_path=png_path,
                            size=args.size,
                            inkscape_path=resolved_inkscape,
                            background=args.background,
                            fit_mode=args.fit_mode,
                        )
                        if args.fit_mode == "pad":
                            width, height = pad_to_square(
                                png_path,
                                size=args.size,
                                background=args.background,
                            )
                        else:
                            width, height = verify_png(png_path)

                        success_count += 1
                        logging.info(
                            "Re-rendered [%s/%s]: %s (%sx%s)",
                            index,
                            total_target,
                            png_path,
                            width,
                            height,
                        )
                        write_jsonl_record(
                            manifest_f,
                            {
                                "source_svg": str(svg_path),
                                "rendered_png": str(png_path),
                                "width": width,
                                "height": height,
                                "status": "re_rendered_corrupt_existing",
                            },
                        )
                else:
                    render_with_inkscape(
                        svg_path=svg_path,
                        png_path=png_path,
                        size=args.size,
                        inkscape_path=resolved_inkscape,
                        background=args.background,
                        fit_mode=args.fit_mode,
                    )
                    if args.fit_mode == "pad":
                        width, height = pad_to_square(
                            png_path,
                            size=args.size,
                            background=args.background,
                        )
                    else:
                        width, height = verify_png(png_path)

                    success_count += 1
                    logging.info(
                        "Rendered [%s/%s]: %s (%sx%s)",
                        index,
                        total_target,
                        png_path,
                        width,
                        height,
                    )
                    write_jsonl_record(
                        manifest_f,
                        {
                            "source_svg": str(svg_path),
                            "rendered_png": str(png_path),
                            "width": width,
                            "height": height,
                            "status": "rendered",
                        },
                    )

            except Exception as exc:
                fail_count += 1
                logging.error("Failed [%s/%s]: %s | %s", index, total_target, svg_path, exc)
                write_jsonl_record(
                    failures_f,
                    {
                        "source_svg": str(svg_path),
                        "error": str(exc),
                        "stage": "render_or_verify",
                    },
                )

            if index % 100 == 0 or index == total_target:
                logging.info(
                    "[%s/%s] rendered=%s skipped=%s failed=%s",
                    index,
                    total_target,
                    success_count,
                    skipped_count,
                    fail_count,
                )

    logging.info("Done.")
    logging.info("Rendered: %s", success_count)
    logging.info("Skipped existing: %s", skipped_count)
    logging.info("Failed: %s", fail_count)
    logging.info("Manifest: %s", manifest_path)
    logging.info("Failures: %s", failures_path)


if __name__ == "__main__":
    main()