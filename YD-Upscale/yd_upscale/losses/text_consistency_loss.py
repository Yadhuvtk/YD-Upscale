from torch import nn  # pyre-ignore[21]
from yd_upscale.utils.registry import LOSSES  # pyre-ignore[21]

@LOSSES.register
class TextConsistencyLoss(nn.Module):
    def __init__(self, loss_weight=1.0):
        super(TextConsistencyLoss, self).__init__()
        self.loss_weight = loss_weight
        
    def forward(self, pred, target):
        return (pred - target).abs().mean() * self.loss_weight
