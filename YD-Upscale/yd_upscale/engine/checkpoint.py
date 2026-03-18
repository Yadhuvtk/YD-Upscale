import torch

def save_checkpoint(model, optimizer, epoch, path):
    state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict() if optimizer else None
    }
    torch.save(state, path)

def load_checkpoint(model, optimizer, path, device):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    if optimizer and checkpoint['optimizer']:
        optimizer.load_state_dict(checkpoint['optimizer'])
    return checkpoint.get('epoch', 0)
