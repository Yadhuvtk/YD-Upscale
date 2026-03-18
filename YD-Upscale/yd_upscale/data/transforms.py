import numpy as np
import torch
import cv2

def img2tensor(img):
    """Convert HWC numpy (RGB, [0, 255]) to CHW torch tensor ([0, 1])."""
    img = img.astype(np.float32) / 255.
    img = torch.from_numpy(np.ascontiguousarray(np.transpose(img, (2, 0, 1))))
    return img
