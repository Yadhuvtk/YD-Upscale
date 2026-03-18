# YD-Upscale

**Founder**: Yadhukrishna  
**Product**: YD-Upscale

A clean, modular, production-ready AI image upscaler project specifically designed for anime, illustrations, and text-preserving artwork. This is a newly branded, original repository structure built from the ground up to support high-quality line preservation, color stability, and text fidelity under synthetic generated degradations.

## Features (v1)
- **Architecture**: RRDB-based generator (inspired by Real-ESRGAN/ESRGAN).
- **Scale**: x4 Upscaling.
- **Degradations**: On-the-fly random blur, noise, resize, and JPEG compression for robust real-world restoration.
- **Losses**: Configurable Charbonnier (pixel) and Edge losses.
- **Metrics**: PSNR and SSIM validation included out-of-the-box (with LPIPS stubbed).

## Installation

```bash
git clone https://github.com/yadhukrishna/YD-Upscale.git
cd YD-Upscale

# Install requirements
pip install -r requirements.txt

# Install as local package
pip install -e .
```

## Dataset Format

Your training images should be high-resolution (HR) ground truths. Place them as follows:
```text
data/
  raw/
    train_hr/   <-- Place train images here
    val_hr/     <-- Place validation images here
    test_hr/    <-- Place test images here
```
*Note: Low-Resolution (LR) images are generated on-the-fly during training.*

## Usage

### Training (Stage 1)
To run the primary training loop with pixel loss:
```bash
python scripts/train_stage1.py -opt configs/train_x4_art_text.yaml
```

### Validation
To evaluate an existing checkpoint on validation data:
```bash
python scripts/validate.py -opt configs/train_x4_art_text.yaml
```

### Inference
To upscale a single image or folder of images:
```bash
python scripts/infer.py -opt configs/infer_x4.yaml
```
*Input imagery should be placed in `data/samples/input` (or modify `infer_x4.yaml`). Outcomes will appear in `outputs/inference`.*

### Benchmarking
Compare model output quantitatively across a test dataset:
```bash
python scripts/benchmark.py -opt configs/benchmark.yaml
```

### Smoke Test
Run a simple integrity check to ensure your environment is fully operational:
```bash
python scripts/quick_test.py
```

## Licensing & Copyright
Please refer to `LICENSE`.
Copyright (c) 2026 Yadhukrishna (YD-Upscale)
