from __future__ import annotations

import random
from io import BytesIO

import cv2
import numpy as np
from PIL import Image


def _pil_to_bgr(img: Image.Image) -> np.ndarray:
    arr = np.array(img.convert("RGB"))
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def _bgr_to_pil(img: np.ndarray) -> Image.Image:
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def degrade_for_illustration(hr: Image.Image, scale: int = 4) -> Image.Image:
    """
    Generate LR from HR on-the-fly for illustration / logo / text training.
    Input: PIL Image
    Output: PIL Image
    """
    img = _pil_to_bgr(hr)
    h, w = img.shape[:2]

    # Ensure valid aligned size first
    h = (h // scale) * scale
    w = (w // scale) * scale
    img = img[:h, :w]

    # Mild blur sometimes
    if random.random() < 0.7:
        sigma = random.uniform(0.6, 1.8)
        k = max(3, int(round(sigma * 3) * 2 + 1))
        img = cv2.GaussianBlur(img, (k, k), sigmaX=sigma)

    # Sometimes add a second resize to create aliasing / resample artifacts
    if random.random() < 0.5:
        tmp_scale = random.uniform(0.65, 0.95)
        tmp_w = max(scale, int(w * tmp_scale))
        tmp_h = max(scale, int(h * tmp_scale))
        img = cv2.resize(img, (tmp_w, tmp_h), interpolation=random.choice([
            cv2.INTER_LINEAR,
            cv2.INTER_AREA,
            cv2.INTER_CUBIC,
        ]))
        img = cv2.resize(img, (w, h), interpolation=random.choice([
            cv2.INTER_LINEAR,
            cv2.INTER_AREA,
            cv2.INTER_CUBIC,
        ]))

    # Main downscale to LR
    lr_w = w // scale
    lr_h = h // scale
    img = cv2.resize(img, (lr_w, lr_h), interpolation=random.choice([
        cv2.INTER_AREA,
        cv2.INTER_LINEAR,
        cv2.INTER_CUBIC,
        cv2.INTER_LANCZOS4,
    ]))

    # Mild noise
    if random.random() < 0.35:
        noise_sigma = random.uniform(1.0, 6.0)
        noise = np.random.normal(0, noise_sigma, img.shape).astype(np.float32)
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # JPEG compression artifacts
    if random.random() < 0.8:
        quality = random.randint(25, 80)
        pil_img = _bgr_to_pil(img)
        buffer = BytesIO()
        pil_img.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        pil_img = Image.open(buffer).convert("RGB")
        img = _pil_to_bgr(pil_img)

    # Optional slight sharpening damage
    if random.random() < 0.25:
        kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]], dtype=np.float32)
        img = cv2.filter2D(img, -1, kernel)

    return _bgr_to_pil(img)