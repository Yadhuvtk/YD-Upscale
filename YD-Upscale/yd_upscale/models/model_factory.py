import torch
from yd_upscale.utils.registry import MODELS

def create_model(opt):
    model_type = opt['type']
    model_class = MODELS.get(model_type)
    if model_class is None:
        raise ValueError(f"Model {model_type} not found in registry.")
    
    # Exclude 'type' from args
    args = {k: v for k, v in opt.items() if k != 'type'}
    
    # Handle pretrained_path if needed
    pretrained_path = args.pop('pretrained_path', None)
    
    model = model_class(**args)
    
    if pretrained_path is not None:
        try:
            checkpoint = torch.load(pretrained_path, map_location='cpu', weights_only=False)
            
            # Check if we need to extract 'params' or 'params_ema'
            state_dict = checkpoint
            if 'params_ema' in checkpoint:
                state_dict = checkpoint['params_ema']
            elif 'params' in checkpoint:
                state_dict = checkpoint['params']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
                
            model.load_state_dict(state_dict, strict=True)
            print(f"Loaded pretrained model from {pretrained_path}")
        except FileNotFoundError:
            print(f"Warning: Pretrained model not found at {pretrained_path}. Using initialized weights.")
        
    return model
