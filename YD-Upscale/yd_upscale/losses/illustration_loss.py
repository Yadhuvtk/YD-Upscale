"""
IllustrationLoss — perceptual loss tuned for anime / flat-colour art.

Components
----------
1. L1 pixel loss
   Simple mean absolute error on RGB values.
   Keeps overall brightness and colour accurate.

2. VGG perceptual loss
   Feature-level L1 loss on intermediate activations of a frozen VGG19.
   Layers used: relu1_2, relu2_2, relu3_4 (progressively deeper features).
   Shallow layers capture lines/textures; deeper layers capture structure/style.
   Weights are tuned so earlier (line-sensitive) layers have higher influence —
   important for anime where sharp outlines matter more than photorealistic texture.

No GAN/adversarial component — this is a stable pre-training / fine-tuning loss.

Usage
-----
    criterion = IllustrationLoss(lambda_l1=1.0, lambda_percep=0.1)
    loss = criterion(sr_batch, hr_batch)   # both in [0, 1], shape (B,3,H,W)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models


# ---------------------------------------------------------------------------
# VGG feature extractor (frozen)
# ---------------------------------------------------------------------------

# Layer names → indices inside vgg19.features for the activations we want.
# relu1_2  → index  3  (after block-1 conv2 + ReLU)
# relu2_2  → index  8  (after block-2 conv2 + ReLU)
# relu3_4  → index 17  (after block-3 conv4 + ReLU)
_VGG_LAYER_INDICES = [3, 8, 17]

# Per-layer perceptual weights: earlier layers (lines) get higher weight for anime
_VGG_LAYER_WEIGHTS = [0.5, 0.25, 0.25]


class VGGFeatureExtractor(nn.Module):
    """
    Extracts intermediate VGG19 features at relu1_2, relu2_2, relu3_4.
    Weights are frozen — we only use the network as a fixed feature space.
    """

    def __init__(self):
        super().__init__()
        # Load pretrained VGG19 (ImageNet weights)
        vgg = tv_models.vgg19(weights=tv_models.VGG19_Weights.IMAGENET1K_V1)

        # Build sub-networks for each slice up to the chosen layer
        features = vgg.features
        self.slices = nn.ModuleList()
        prev_end = 0
        for idx in _VGG_LAYER_INDICES:
            self.slices.append(nn.Sequential(*[features[i] for i in range(prev_end, idx + 1)]))
            prev_end = idx + 1

        # Freeze all parameters — VGG is used only as a fixed feature extractor
        for param in self.parameters():
            param.requires_grad = False

        # ImageNet normalisation constants (input expected in [0, 1])
        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
        )

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalise from [0,1] to ImageNet mean/std."""
        return (x - self.mean) / self.std

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """
        Args:
            x: (B, 3, H, W) in [0, 1]
        Returns:
            List of feature maps, one per chosen layer.
        """
        x = self._normalize(x)
        feats = []
        for s in self.slices:
            x = s(x)
            feats.append(x)
        return feats


# ---------------------------------------------------------------------------
# Perceptual Loss
# ---------------------------------------------------------------------------

class VGGPerceptualLoss(nn.Module):
    """
    Weighted L1 distance in VGG feature space.
    """

    def __init__(self):
        super().__init__()
        self.extractor = VGGFeatureExtractor()
        self.weights = _VGG_LAYER_WEIGHTS

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred, target: (B, 3, H, W) float tensors in [0, 1]
        Returns:
            Scalar perceptual loss.
        """
        # Stop gradients through the target path — no need to backprop into target
        with torch.no_grad():
            target_feats = self.extractor(target.detach())

        pred_feats = self.extractor(pred)

        loss = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
        for w, pf, tf in zip(self.weights, pred_feats, target_feats):
            loss = loss + w * F.l1_loss(pf, tf)
        return loss


# ---------------------------------------------------------------------------
# Combined Illustration Loss
# ---------------------------------------------------------------------------

class IllustrationLoss(nn.Module):
    """
    Combined loss for anime / illustration super-resolution.

        loss = lambda_l1 * L1(pred, target)
             + lambda_percep * VGG_perceptual(pred, target)

    Default weights (lambda_l1=1.0, lambda_percep=0.1) keep pixel accuracy
    dominant while letting perceptual loss improve sharpness and line quality.

    Args:
        lambda_l1     : weight on pixel-wise L1 loss (default 1.0)
        lambda_percep : weight on VGG perceptual loss (default 0.1)
    """

    def __init__(self, lambda_l1: float = 1.0, lambda_percep: float = 0.1):
        super().__init__()
        self.lambda_l1     = lambda_l1
        self.lambda_percep = lambda_percep
        self.perceptual    = VGGPerceptualLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred   : (B, 3, H, W) network output, values in [0, 1]
            target : (B, 3, H, W) ground-truth HR image, values in [0, 1]
        Returns:
            Scalar combined loss.
        """
        l1_loss      = F.l1_loss(pred, target)
        percep_loss  = self.perceptual(pred, target)

        return self.lambda_l1 * l1_loss + self.lambda_percep * percep_loss
