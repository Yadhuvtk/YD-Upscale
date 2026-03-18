from yd_upscale.utils.registry import LOSSES

def build_loss(opt):
    loss_type = opt['type']
    loss_class = LOSSES.get(loss_type)
    if loss_class is None:
        raise ValueError(f"Loss {loss_type} not found in registry.")
    
    args = {k: v for k, v in opt.items() if k != 'type'}
    return loss_class(**args)
