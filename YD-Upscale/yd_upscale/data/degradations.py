import random
import cv2
import numpy as np

def apply_blur(img, sigma_range):
    if random.random() < 0.5:
        sigma = random.uniform(sigma_range[0], sigma_range[1])
        k_size = int(np.ceil(sigma * 3)) * 2 + 1
        img = cv2.GaussianBlur(img, (k_size, k_size), sigmaX=sigma, sigmaY=sigma)
    return img

def apply_noise(img, noise_range):
    noise_level = random.uniform(noise_range[0], noise_range[1])
    noise = np.random.normal(0, noise_level, img.shape).astype(np.float32)
    img_noisy = img.astype(np.float32) + noise
    return np.clip(img_noisy, 0, 255).astype(np.uint8)

def apply_jpeg(img, quality_range):
    quality = random.randint(quality_range[0], quality_range[1])
    _, encimg = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    img = cv2.imdecode(encimg, cv2.IMREAD_UNCHANGED)
    return img

def degrade_image(img_hr, degradations_opt, scale):
    """Generates LR from HR"""
    img_lr = img_hr.copy()
    
    # Optional blur
    if random.random() < degradations_opt.get('blur_prob', 0.5):
        img_lr = apply_blur(img_lr, degradations_opt.get('blur_sigma', [0.2, 1.5]))
        
    # Resize down
    h, w = img_lr.shape[:2]
    img_lr = cv2.resize(img_lr, (w // scale, h // scale), interpolation=cv2.INTER_CUBIC)
    
    # Optional noise
    if random.random() < degradations_opt.get('noise_prob', 0.5):
        img_lr = apply_noise(img_lr, degradations_opt.get('noise_range', [1.0, 15.0]))
        
    # Optional JPEG
    if random.random() < degradations_opt.get('jpeg_prob', 0.8):
        img_lr = apply_jpeg(img_lr, degradations_opt.get('jpeg_range', [60, 95]))
        
    return img_lr
