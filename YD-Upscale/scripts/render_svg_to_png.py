from __future__ import annotations

import argparse
import json
import logging
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


def render_with_inkscape(
    svg_path: Path,
    png_path: Path,
    size: int,
    inkscape_path: str = "inkscape",
    background: str = "white",
) -> None:
    png_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        inkscape_path,
        str(svg_path),
        "--export-type=png",
        f"--export-filename={png_path}",
        f"--export-width={size}",
        f"--export-height={size}",
    ]

    if background == "white":
        cmd.extend(["--export-background=white", "--export-background-opacity=1.0"])
    elif background == "transparent":
        cmd.extend(["--export-background-opacity=0.0"])

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
    )

    if result.returncode != 0:
        stderr = result.stderr.strip() or "Unknown Inkscape error"
        raise RuntimeError(stderr)


def write_jsonl_record(file_obj, record: dict) -> None:
    file_obj.write(json.dumps(record, ensure_ascii=False) + "\n")


def iter_svg_files(files: list[Path], limit: int | None) -> Iterable[Path]:
    if limit is None:
        yield from files
    else:
        yield from files[:limit]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render SVG files into PNG images for YD-Upscale dataset preparation."
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
        help="Square PNG output size. Default: 2048",
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

            try:
                if png_path.exists() and not args.overwrite:
                    try:
                        width, height = verify_png(png_path)
                        skipped_count += 1
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
                            inkscape_path=args.inkscape_path,
                            background=args.background,
                        )
                        width, height = verify_png(png_path)
                        success_count += 1
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
                        inkscape_path=args.inkscape_path,
                        background=args.background,
                    )
                    width, height = verify_png(png_path)
                    success_count += 1
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
                write_jsonl_record(
                    failures_f,
                    {
                        "source_svg": str(svg_path),
                        "error": str(exc),
                        "stage": "render_or_verify",
                    },
                )
                logging.error("Failed: %s | %s", svg_path, exc)

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