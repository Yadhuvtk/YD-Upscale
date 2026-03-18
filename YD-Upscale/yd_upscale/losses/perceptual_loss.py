from torch import nn
from yd_upscale.utils.registry import LOSSES

@LOSSES.register
class PerceptualLoss(nn.Module):
    def __init__(self, layer_weights=None, vgg_type='vgg19', use_input_norm=True, perceptual_weight=1.0, style_weight=0):
        super(PerceptualLoss, self).__init__()
        self.perceptual_weight = perceptual_weight
        self.style_weight = style_weight
    
    def forward(self, pred, target):
        # Stub
        return pred.sum() * 0.0
