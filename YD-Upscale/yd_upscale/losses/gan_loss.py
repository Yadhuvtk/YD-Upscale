from torch import nn
from yd_upscale.utils.registry import LOSSES

@LOSSES.register
class GANLoss(nn.Module):
    def __init__(self, gan_type, real_label_val=1.0, fake_label_val=0.0, loss_weight=1.0):
        super(GANLoss, self).__init__()
        self.loss_weight = loss_weight
        
    def forward(self, pred, target_is_real):
        # Stub
        return pred.sum() * 0.0
