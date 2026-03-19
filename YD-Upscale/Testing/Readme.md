# 🚀 YD-Upscale (RRDB x4)

**Author:** Yadhukrishna  
**Contact:** yadhuvtk@gmail.com  
**Model Type:** Super Resolution (x4)  
**Architecture:** RRDBNet (ESRGAN-style)  

---

## 🔥 Overview

**YD-Upscale** is a deep learning–based super-resolution model designed specifically for:

- Line-art
- Logos
- Ornamental designs
- SVG-derived raster images
- Vector-like illustrations

It enhances low-resolution images by **upscaling 4× (x4)** while preserving sharp edges, clean curves, and structural clarity.

---

## 🎯 Key Features

- ✨ Sharp edge reconstruction  
- ✨ Smooth curve enhancement  
- ✨ Clean line-art upscaling  
- ✨ Optimized for vector-style graphics  
- ✨ Minimal artifacts on structured designs  

---

## 🧠 Model Details

- **Architecture:** RRDBNet (Residual-in-Residual Dense Blocks)  
- **Scale Factor:** x4  
- **Training Data:**
  - SVG → high-resolution raster (2048px)
  - Synthetic degraded LR (x4)
- **Training Type:** Supervised (L1 loss)  
- **Patch Training:** 128 → 512  

---

## ✅ Best Use Cases

This model performs best on:

- Logos  
- Icons  
- Line drawings  
- Decorative patterns  
- Engraving-style artwork  
- Flat illustrations  

---

## ⚠️ Limitations

This model is **not designed for:**

- Human faces  
- Natural photographs  
- Highly textured images  
- Real-world noisy images  

---

## 📦 Installation

```bash
pip install torch torchvision pillow safetensors