import numpy as np
from skimage.metrics import structural_similarity as ssim

def calculate_ssim(img1, img2, crop_border=0):
    if crop_border != 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]
    
    ssim_val = ssim(img1, img2, data_range=255, multichannel=True, channel_axis=-1)
    return ssim_val
