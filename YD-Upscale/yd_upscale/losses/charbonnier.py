import torch
from torch import nn
from yd_upscale.utils.registry import LOSSES

@LOSSES.register
class CharbonnierLoss(nn.Module):
    def __init__(self, loss_weight=1.0, eps=1e-6):
        super(CharbonnierLoss, self).__init__()
        self.loss_weight = loss_weight
        self.eps = eps

    def forward(self, pred, target):
        loss = torch.sqrt((pred - target)**2 + self.eps)
        return torch.mean(loss) * self.loss_weight
