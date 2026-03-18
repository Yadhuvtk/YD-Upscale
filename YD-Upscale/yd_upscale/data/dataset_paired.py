import torch
from torch.utils.data import Dataset
from pathlib import Path
import random
import cv2

from yd_upscale.utils.registry import DATASETS
from yd_upscale.utils.image_io import read_image
from .degradations import degrade_image
from .transforms import img2tensor

@DATASETS.register
class PairedDataset(Dataset):
    def __init__(self, opt):
        self.opt = opt
        self.hr_paths = []
        
        manifest_file = opt.get('manifest_file')
        if manifest_file and Path(manifest_file).exists():
            with open(manifest_file, 'r') as f:
                self.hr_paths = [Path(opt['dataroot_hr']) / line.strip() for line in f if line.strip()]
        else:
            self.hr_paths = [p for p in sorted(list(Path(opt['dataroot_hr']).glob('*.*'))) if p.suffix.lower() in ['.png', '.jpg', '.jpeg', '.webp']]
            
        self.patch_size = opt.get('patch_size', 256)
        # Using 4 as default, in a broader scope pass from model_opt
        self.scale = 4 
        self.degradations = opt.get('degradations', {})

    def __len__(self):
        return max(1, len(self.hr_paths))  # avoid 0 len for empty dirs during stubs

    def __getitem__(self, index):
        if not self.hr_paths:
            # Return dummy tensor if folder is completely empty during scaffold init
            return {
                'lr': torch.zeros(3, 64, 64),
                'hr': torch.zeros(3, 256, 256),
                'hr_path': 'dummy.png'
            }

        hr_path = self.hr_paths[index]
        img_hr = read_image(hr_path)
        
        if self.patch_size:
            h, w = img_hr.shape[:2]
            if h < self.patch_size or w < self.patch_size:
                img_hr = cv2.resize(img_hr, (max(self.patch_size, w), max(self.patch_size, h)))
                h, w = img_hr.shape[:2]

            top = random.randint(0, h - self.patch_size)
            left = random.randint(0, w - self.patch_size)
            img_hr = img_hr[top:top + self.patch_size, left:left + self.patch_size, :]
            
        img_lr = degrade_image(img_hr, self.degradations, self.scale)
        
        img_hr = img2tensor(img_hr)
        img_lr = img2tensor(img_lr)

        return {'lr': img_lr, 'hr': img_hr, 'hr_path': str(hr_path)}
