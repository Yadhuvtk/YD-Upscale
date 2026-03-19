from __future__ import annotations

from pathlib import Path

import torch
from safetensors.torch import save_file


def main():
    project_root = Path(r"E:\Yadhu Projects\YD-Upscale\YD-Upscale")

    input_ckpt = project_root / "checkpoints" / "yd_upscale_rrdb_x4_epoch_1.pth"
    output_file = project_root / "checkpoints" / "YD-Upscale.safetensors"

    if not input_ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {input_ckpt}")

    ckpt = torch.load(input_ckpt, map_location="cpu")

    if "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    else:
        state_dict = ckpt

    safe_state_dict = {}
    for k, v in state_dict.items():
        if isinstance(v, torch.Tensor):
            safe_state_dict[k] = v.detach().contiguous().cpu()

    metadata = {
        "format": "pt",
        "model": "YD-Upscale RRDB x4",
        "scale": "4",
        "author": "Yadhukrishna",
    }

    save_file(safe_state_dict, str(output_file), metadata=metadata)

    print(f"Saved safetensors file to: {output_file}")
    print(f"Tensor count: {len(safe_state_dict)}")


if __name__ == "__main__":
    main()