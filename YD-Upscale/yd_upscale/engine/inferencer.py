import torch
import cv2
import numpy as np
from pathlib import Path

from yd_upscale.utils.device import get_device
from yd_upscale.utils.image_io import read_image, save_image
from yd_upscale.data.transforms import img2tensor

class Inferencer:
    def __init__(self, model, device=None, use_half=False):
        self.model = model
        self.device = device or get_device()
        self.use_half = use_half
        self.model.eval()
        if self.use_half:
            self.model.half()

    @torch.no_grad()
    def infer(self, img_path, out_path):
        img = read_image(img_path)
        tensor = img2tensor(img).unsqueeze(0).to(self.device)
        
        if self.use_half:
            tensor = tensor.half()
            
        pred = self.model(tensor)
        
        pred = pred.squeeze(0).float().cpu().numpy()
        pred = np.clip(np.transpose(pred, (1, 2, 0)) * 255.0, 0, 255).astype(np.uint8)
        
        save_image(pred, out_path)
