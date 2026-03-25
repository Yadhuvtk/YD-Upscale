"""
inference.py — 4× anime upscaling with YD-Upscale.

Loads EMA weights (params_ema key) from a trained checkpoint and upscales
images using tiled inference to avoid OOM on large inputs.

Modes
-----
  Single image:
    python inference.py -i photo.jpg -o result.png

  Batch folder:
    python inference.py -i input_folder/ -o output_folder/

  Custom tile / checkpoint:
    python inference.py -i img.jpg --tile 256 --tile-pad 16 \
        -c checkpoints/YD_UPSCALE_anime_x4_iter_200000.pth

  Disable tiling (small images only):
    python inference.py -i small.png --no-tile
"""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import torch
import torchvision.transforms.functional as TF
from PIL import Image

from yd_upscale.models.rrdbnet import RRDBNet

# Supported image extensions for batch mode
_IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


# ============================================================================
# Tiled inference
# ============================================================================

def upscale_tiled(
    model: torch.nn.Module,
    lr: torch.Tensor,
    scale: int = 4,
    tile: int = 512,
    tile_pad: int = 32,
) -> torch.Tensor:
    """
    Process a large LR image in overlapping tiles.

    1. Divide LR into tile×tile blocks with tile_pad overlap on each side.
    2. Run each padded tile through the model.
    3. Crop out the padding from the SR output.
    4. Stitch into the full SR canvas.

    The overlap prevents visible seam artefacts at tile boundaries.

    Args:
        model    : RRDBNet in eval mode
        lr       : (1, 3, H, W) input tensor, [0,1]
        scale    : upscale factor
        tile     : tile size in LR pixels
        tile_pad : overlap padding in LR pixels

    Returns:
        (1, 3, H*scale, W*scale) tensor, [0,1]
    """
    _, _, h, w = lr.shape
    out_h, out_w = h * scale, w * scale

    output = torch.zeros(1, 3, out_h, out_w, dtype=lr.dtype, device=lr.device)

    tiles_y = math.ceil(h / tile)
    tiles_x = math.ceil(w / tile)

    for ty in range(tiles_y):
        for tx in range(tiles_x):
            # LR tile boundaries
            x0 = tx * tile
            y0 = ty * tile
            x1 = min(x0 + tile, w)
            y1 = min(y0 + tile, h)

            # Add padding (clipped to image bounds)
            px0 = max(x0 - tile_pad, 0)
            py0 = max(y0 - tile_pad, 0)
            px1 = min(x1 + tile_pad, w)
            py1 = min(y1 + tile_pad, h)

            # Run model on padded tile
            lr_tile = lr[:, :, py0:py1, px0:px1]
            with torch.no_grad():
                sr_tile = model(lr_tile).clamp(0.0, 1.0)

            # Crop padding from the SR output
            left_pad = (x0 - px0) * scale
            top_pad  = (y0 - py0) * scale
            crop_w   = (x1 - x0) * scale
            crop_h   = (y1 - y0) * scale

            sr_crop = sr_tile[:, :, top_pad:top_pad + crop_h, left_pad:left_pad + crop_w]

            # Place into output canvas
            ox = x0 * scale
            oy = y0 * scale
            output[:, :, oy:oy + crop_h, ox:ox + crop_w] = sr_crop

    return output


# ============================================================================
# Full-image inference
# ============================================================================

def upscale_full(model: torch.nn.Module, lr: torch.Tensor) -> torch.Tensor:
    """Single forward pass — only for images that fit in VRAM."""
    with torch.no_grad():
        return model(lr).clamp(0.0, 1.0)


# ============================================================================
# Model loader
# ============================================================================

def load_model(ckpt_path: Path, device: torch.device, scale: int = 4) -> RRDBNet:
    """
    Load RRDBNet anime-6B and populate weights from checkpoint.
    Prefers params_ema → model_state_dict → raw state_dict.
    """
    print(f"Loading: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    if isinstance(ckpt, dict):
        if "params_ema" in ckpt:
            state = ckpt["params_ema"]
            print("  Using params_ema weights.")
        elif "model_state_dict" in ckpt:
            state = ckpt["model_state_dict"]
            print("  Using model_state_dict (no EMA found).")
        elif "params" in ckpt:
            state = ckpt["params"]
            print("  Using params key.")
        else:
            state = ckpt
            print("  Treating checkpoint as raw state_dict.")
    else:
        state = ckpt

    model = RRDBNet(in_nc=3, out_nc=3, nf=64, nb=6, gc=32, scale=scale)
    model.load_state_dict(state, strict=True)
    model.eval().to(device)
    return model


# ============================================================================
# Single-image upscale
# ============================================================================

def upscale_image(
    model: torch.nn.Module,
    input_path: Path,
    output_path: Path,
    device: torch.device,
    scale: int,
    tile: int,
    tile_pad: int,
    use_tile: bool,
    use_fp16: bool,
) -> None:
    """Load one image, upscale, save."""
    img = Image.open(input_path).convert("RGB")
    lr = TF.to_tensor(img).unsqueeze(0)

    if use_fp16:
        lr = lr.half()
    lr = lr.to(device)

    t0 = time.time()
    if use_tile:
        sr = upscale_tiled(model, lr, scale=scale, tile=tile, tile_pad=tile_pad)
    else:
        sr = upscale_full(model, lr)
    elapsed = time.time() - t0

    sr = sr.squeeze(0).float().cpu().clamp(0.0, 1.0)
    TF.to_pil_image(sr).save(output_path)

    print(
        f"  {input_path.name} ({img.width}x{img.height}) → "
        f"{output_path.name} ({img.width * scale}x{img.height * scale})  "
        f"[{elapsed:.1f}s]"
    )


# ============================================================================
# Args
# ============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="YD-Upscale anime 4x inference")

    p.add_argument("-i", "--input", required=True,
                   help="Input image path OR folder for batch mode")
    p.add_argument("-o", "--output", default=None,
                   help="Output path/folder. Default: <stem>_x4.png or output_x4/")
    p.add_argument("-c", "--checkpoint",
                   default="checkpoints/YD_UPSCALE_anime_x4_best.pth",
                   help="Checkpoint .pth path")
    p.add_argument("--scale",    type=int, default=4)
    p.add_argument("--tile",     type=int, default=512,
                   help="Tile size in LR pixels (default 512)")
    p.add_argument("--tile-pad", type=int, default=32,
                   help="Tile overlap padding in LR pixels (default 32)")
    p.add_argument("--no-tile",  action="store_true",
                   help="Disable tiling (single forward pass)")
    p.add_argument("--fp32",     action="store_true",
                   help="Force float32 on GPU (default: fp16)")
    return p.parse_args()


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    args = parse_args()

    input_path = Path(args.input)
    ckpt_path  = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_fp16 = (device.type == "cuda") and (not args.fp32)
    use_tile = (not args.no_tile) and (args.tile > 0)
    print(f"Device: {device}  |  FP16: {use_fp16}  |  Tile: {args.tile if use_tile else 'off'}")

    model = load_model(ckpt_path, device, scale=args.scale)
    if use_fp16:
        model = model.half()

    # ------------------------------------------------------------------
    # Batch mode: input is a folder
    # ------------------------------------------------------------------
    if input_path.is_dir():
        images = sorted(
            p for p in input_path.iterdir()
            if p.suffix.lower() in _IMG_EXTS
        )
        if not images:
            print(f"No images found in {input_path}")
            return

        if args.output:
            out_dir = Path(args.output)
        else:
            out_dir = input_path.parent / f"{input_path.name}_x{args.scale}"
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"Batch: {len(images)} images → {out_dir}")
        for img_path in images:
            out_path = out_dir / f"{img_path.stem}_x{args.scale}.png"
            upscale_image(
                model, img_path, out_path, device,
                args.scale, args.tile, args.tile_pad, use_tile, use_fp16,
            )
        print("Batch complete.")

    # ------------------------------------------------------------------
    # Single image mode
    # ------------------------------------------------------------------
    else:
        if not input_path.exists():
            raise FileNotFoundError(f"Input not found: {input_path}")

        if args.output:
            output_path = Path(args.output)
        else:
            output_path = input_path.parent / f"{input_path.stem}_x{args.scale}.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        upscale_image(
            model, input_path, output_path, device,
            args.scale, args.tile, args.tile_pad, use_tile, use_fp16,
        )
    print("Done.")


if __name__ == "__main__":
    main()
