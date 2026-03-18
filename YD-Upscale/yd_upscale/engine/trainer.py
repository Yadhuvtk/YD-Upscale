import torch  # pyre-ignore[21]
import torch.nn as nn  # pyre-ignore[21]
from yd_upscale.utils.device import get_device  # pyre-ignore[21]

class Trainer:
    def __init__(self, model, optim_g, losses, device=None, use_amp=False, resume_state=None):
        self.model = model
        self.optim_g = optim_g
        self.losses = losses
        self.device = device or get_device()
        self.use_amp = use_amp
        
        # Determine appropriate amp scaler class
        try:
            from torch.amp import GradScaler  # pyre-ignore[21]
            self.scaler = GradScaler('cuda', enabled=self.use_amp)
        except ImportError:
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        
        self.start_epoch = 0
        if resume_state:
            self._resume(resume_state)

    def _resume(self, resume_state_path):
        loc = f'cuda:{torch.cuda.current_device()}' if torch.cuda.is_available() else 'cpu'
        checkpoint = torch.load(resume_state_path, map_location=loc, weights_only=False)
        self.model.load_state_dict(checkpoint['state_dict'])
        if self.optim_g and 'optimizer' in checkpoint:
            self.optim_g.load_state_dict(checkpoint['optimizer'])
        self.start_epoch = checkpoint.get('epoch', 0)
        print(f"Resumed training from epoch {self.start_epoch}")

    def train_step(self, data):
        self.model.train()
        self.optim_g.zero_grad()
        
        lr = data['lr'].to(self.device)
        hr = data['hr'].to(self.device)
        
        if self.use_amp:
            try:
                from torch.amp import autocast  # pyre-ignore[21]
                ctx = autocast('cuda', enabled=True)
            except ImportError:
                ctx = torch.cuda.amp.autocast(enabled=True)
            
            with ctx:
                pred = self.model(lr)
                loss_dict = {}
                total_loss = 0
                
                # calculate losses
                for name, criterion in self.losses.items():
                    l = criterion(pred, hr)
                    loss_dict[name] = l
                    total_loss += l
        else:
            pred = self.model(lr)
            loss_dict = {}
            total_loss = 0
            for name, criterion in self.losses.items():
                l = criterion(pred, hr)
                loss_dict[name] = l
                total_loss += l
                
        self.scaler.scale(total_loss).backward()
        self.scaler.step(self.optim_g)
        self.scaler.update()
        
        # return detached losses for logging
        return {k: v.item() for k, v in loss_dict.items()}
