import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from yd_upscale.losses.charbonnier import CharbonnierLoss
import torch

def test_charbonnier():
    loss_fn = CharbonnierLoss()
    pred = torch.zeros(1, 3, 16, 16)
    target = torch.ones(1, 3, 16, 16)
    loss = loss_fn(pred, target)
    assert loss.item() > 0.99 # Should be ~1.0
