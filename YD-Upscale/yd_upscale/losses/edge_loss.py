import torch
from torch import nn
import torch.nn.functional as F
from yd_upscale.utils.registry import LOSSES

@LOSSES.register
class EdgeLoss(nn.Module):
    def __init__(self, loss_weight=1.0, edge_type='sobel'):
        super(EdgeLoss, self).__init__()
        self.loss_weight = loss_weight
        self.edge_type = edge_type
        
        if edge_type == 'sobel':
            # Create sobel filters
            k_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3)
            k_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).view(1, 1, 3, 3)
            self.register_buffer('weight_x', k_x)
            self.register_buffer('weight_y', k_y)
        
    def forward(self, pred, target):
        if self.edge_type == 'sobel':
            b, c, h, w = pred.shape
            
            # Combine channels for generic edge detection if needed, or apply per channel
            pred_edges = self._compute_sobel(pred.view(-1, 1, h, w))
            target_edges = self._compute_sobel(target.view(-1, 1, h, w))
            loss = F.l1_loss(pred_edges, target_edges)
        else:
            loss = F.l1_loss(pred, target)
        
        return loss * self.loss_weight
        
    def _compute_sobel(self, x):
        grad_x = F.conv2d(x, self.weight_x, padding=1)
        grad_y = F.conv2d(x, self.weight_y, padding=1)
        return torch.sqrt(grad_x**2 + grad_y**2 + 1e-6)
