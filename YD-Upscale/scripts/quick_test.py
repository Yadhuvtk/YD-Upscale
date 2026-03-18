import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
from yd_upscale.models.rrdbnet import RRDBNet

def main():
    print("Initializing RRDBNet...")
    model = RRDBNet(3, 3, 64, 23)
    dummy_input = torch.randn(1, 3, 64, 64)
    print("Running dummy forward pass...")
    out = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {out.shape}")
    assert out.shape == (1, 3, 256, 256), "Output shape should be x4 upscale"
    print("Smoke test passed!")

if __name__ == '__main__':
    main()
