import torch
from torch import nn
from yd_upscale.utils.registry import MODELS

@MODELS.register
class UNetDiscriminatorSN(nn.Module):
    """Defines a U-Net discriminator with spectral normalization.
    Stub for future stage 2.
    """
    def __init__(self, num_in_ch=3, num_feat=64):
        super(UNetDiscriminatorSN, self).__init__()
        self.dummy_param = nn.Parameter(torch.empty(0))

    def forward(self, x):
        pass
