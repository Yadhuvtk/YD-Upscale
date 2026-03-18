import torch
from torch import nn
from yd_upscale.utils.registry import LOSSES

@LOSSES.register
class L1Loss(nn.Module):
    def __init__(self, loss_weight=1.0):
        super(L1Loss, self).__init__()
        self.loss_weight = loss_weight

    def forward(self, pred, target):
        return torch.mean(torch.abs(pred - target)) * self.loss_weight

@LOSSES.register
class MSELoss(nn.Module):
    def __init__(self, loss_weight=1.0):
        super(MSELoss, self).__init__()
        self.loss_weight = loss_weight

    def forward(self, pred, target):
        return torch.mean((pred - target)**2) * self.loss_weight
