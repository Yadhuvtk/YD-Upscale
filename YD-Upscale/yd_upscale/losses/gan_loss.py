"""
GAN Loss module for Real-ESRGAN–style adversarial training.

Combines three loss components:
  1. L1 pixel loss         (weight 1.0) — keeps colours/brightness accurate
  2. VGG perceptual loss   (weight 1.0) — matches feature-level structure/texture
  3. Vanilla GAN loss      (weight 0.1) — adversarial signal from discriminator

The VGG feature extractor uses a frozen VGG19 and taps five layers:
    conv1_2, conv2_2, conv3_4, conv4_4, conv5_4
with weights [0.1, 0.1, 1.0, 1.0, 1.0] — exactly matching the Real-ESRGAN
perceptual loss configuration (before-activation features).

Usage
-----
    # Build loss
    criterion = GeneratorLoss()                   # on the same device as model
    d_criterion = DiscriminatorLoss()

    # Generator step
    g_loss, g_log = criterion(sr, hr, disc_fake_pred)

    # Discriminator step
    d_loss, d_log = d_criterion(disc_real_pred, disc_fake_pred)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models


# ============================================================================
# VGG19 Feature Extractor (frozen)
# ============================================================================

# Indices into vgg19.features where we extract activations (before ReLU):
#   conv1_2=2, conv2_2=7, conv3_4=16, conv4_4=25, conv5_4=34
# Using before-activation features matches Real-ESRGAN's configuration.
_VGG_LAYER_INDICES = [2, 7, 16, 25, 34]
_VGG_LAYER_WEIGHTS = [0.1, 0.1, 1.0, 1.0, 1.0]


class VGGFeatureExtractor(nn.Module):
    """
    Extracts intermediate VGG19 features for perceptual loss computation.
    All weights are frozen — the VGG serves as a fixed feature space only.
    """

    def __init__(self):
        super().__init__()
        vgg = tv_models.vgg19(weights=tv_models.VGG19_Weights.IMAGENET1K_V1)
        features = vgg.features

        # Build sequential slices that output features at each target layer
        self.slices = nn.ModuleList()
        prev = 0
        for idx in _VGG_LAYER_INDICES:
            self.slices.append(nn.Sequential(*[features[i] for i in range(prev, idx + 1)]))
            prev = idx + 1

        # Freeze everything
        for p in self.parameters():
            p.requires_grad = False

        # ImageNet normalisation (input expected in [0, 1])
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std",  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        x = (x - self.mean) / self.std
        feats = []
        for s in self.slices:
            x = s(x)
            feats.append(x)
        return feats


# ============================================================================
# Perceptual Loss (L1 in VGG feature space)
# ============================================================================

class VGGPerceptualLoss(nn.Module):
    """Weighted L1 distance across multiple VGG19 feature layers."""

    def __init__(self):
        super().__init__()
        self.extractor = VGGFeatureExtractor()
        self.layer_weights = _VGG_LAYER_WEIGHTS

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            target_feats = self.extractor(target.detach())
        pred_feats = self.extractor(pred)

        loss = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
        for w, pf, tf in zip(self.layer_weights, pred_feats, target_feats):
            loss = loss + w * F.l1_loss(pf, tf)
        return loss


# ============================================================================
# Vanilla GAN Loss (BCE with logits)
# ============================================================================

def _gan_loss(pred: torch.Tensor, target_is_real: bool) -> torch.Tensor:
    """
    Vanilla GAN loss using binary cross-entropy with logits.
    - For real targets:  -log(sigmoid(pred))
    - For fake targets:  -log(1 - sigmoid(pred))
    """
    target = torch.ones_like(pred) if target_is_real else torch.zeros_like(pred)
    return F.binary_cross_entropy_with_logits(pred, target)


# ============================================================================
# Generator Loss (L1 + Perceptual + GAN)
# ============================================================================

class GeneratorLoss(nn.Module):
    """
    Combined generator loss for GAN training.

        total = lambda_l1 * L1(sr, hr)
              + lambda_percep * VGGPercep(sr, hr)
              + lambda_gan * GAN(disc_pred, real=True)

    Args:
        lambda_l1      : pixel-loss weight   (default 1.0)
        lambda_percep  : perceptual weight   (default 1.0)
        lambda_gan     : adversarial weight  (default 0.1)
    """

    def __init__(
        self,
        lambda_l1: float = 1.0,
        lambda_percep: float = 1.0,
        lambda_gan: float = 0.1,
    ):
        super().__init__()
        self.lambda_l1     = lambda_l1
        self.lambda_percep = lambda_percep
        self.lambda_gan    = lambda_gan
        self.perceptual    = VGGPerceptualLoss()

    def forward(
        self,
        sr: torch.Tensor,
        hr: torch.Tensor,
        disc_fake_pred: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Args:
            sr             : (B,3,H,W) generator output
            hr             : (B,3,H,W) ground-truth HR
            disc_fake_pred : discriminator output on sr (None during warmup phase)

        Returns:
            total_loss : scalar
            log_dict   : individual loss values for logging
        """
        # L1 pixel loss
        l1 = F.l1_loss(sr, hr)

        # VGG perceptual loss
        percep = self.perceptual(sr, hr)

        total = self.lambda_l1 * l1 + self.lambda_percep * percep

        log = {
            "g_l1": l1.item(),
            "g_percep": percep.item(),
        }

        # Adversarial loss — only when discriminator is active
        if disc_fake_pred is not None:
            gan = _gan_loss(disc_fake_pred, target_is_real=True)
            total = total + self.lambda_gan * gan
            log["g_gan"] = gan.item()

        log["g_total"] = total.item()
        return total, log


# ============================================================================
# Discriminator Loss
# ============================================================================

class DiscriminatorLoss(nn.Module):
    """
    Standard discriminator loss: push real→1, fake→0.

        loss = GAN(real_pred, real=True) + GAN(fake_pred, real=False)
    """

    def forward(
        self,
        real_pred: torch.Tensor,
        fake_pred: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        loss_real = _gan_loss(real_pred, target_is_real=True)
        loss_fake = _gan_loss(fake_pred, target_is_real=False)
        total = (loss_real + loss_fake) / 2.0

        log = {
            "d_real": loss_real.item(),
            "d_fake": loss_fake.item(),
            "d_total": total.item(),
        }
        return total, log
