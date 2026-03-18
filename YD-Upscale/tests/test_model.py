import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from yd_upscale.models.rrdbnet import RRDBNet
import torch

def test_rrdbnet():
    model = RRDBNet(3, 3, 64, 23)
    x = torch.randn(1, 3, 32, 32)
    out = model(x)
    assert out.shape == (1, 3, 128, 128)
