from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
import torchvision.transforms.functional as TF

from yd_upscale.models.rrdbnet import RRDBNet


def main():
    testing_dir = Path(__file__).resolve().parent
    project_root = testing_dir.parent

    checkpoint_path = project_root / "checkpoints" / "yd_upscale_rrdb_x4_epoch_1.pth"
    input_path = testing_dir / "Tests.jpg"
    output_dir = testing_dir / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "test_sr_x4_epoch1.png"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    if not input_path.exists():
        raise FileNotFoundError(f"Input image not found: {input_path}")

    model = RRDBNet(
        in_nc=3,
        out_nc=3,
        nf=64,
        nb=23,
        gc=32,
        scale=4,
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    img = Image.open(input_path).convert("RGB")
    lr_tensor = TF.to_tensor(img).unsqueeze(0).to(device)

    with torch.no_grad():
        sr_tensor = model(lr_tensor).clamp(0, 1)

    sr_img = TF.to_pil_image(sr_tensor.squeeze(0).cpu())
    sr_img.save(output_path)

    print(f"Saved output to: {output_path}")
    print(f"Output size: {sr_img.size}")


if __name__ == "__main__":
    main()