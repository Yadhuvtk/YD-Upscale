from torch.optim.lr_scheduler import MultiStepLR

def build_scheduler(opt, optimizer):
    if opt['type'] == 'MultiStepLR':
        return MultiStepLR(optimizer, milestones=opt['milestones'], gamma=opt['gamma'])
    else:
        # Default placeholder to prevent crashes
        return MultiStepLR(optimizer, milestones=[1000000], gamma=1.0)
