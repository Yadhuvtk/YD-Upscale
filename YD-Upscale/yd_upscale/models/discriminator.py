"""
U-Net Discriminator with Spectral Normalization.

This is the same architecture used in Real-ESRGAN for GAN-based training.
It provides both global (encoder path) and local (decoder path) feedback
to the generator, which helps produce sharp details without introducing
checkerboard/ringing artefacts.

Architecture
------------
Encoder (downsampling path):
    conv0 (3→64)  → conv1 (64→128,  stride=2) → conv2 (128→256,  stride=2)
    → conv3 (256→512, stride=2) → conv4 (512→512, stride=2)
    → conv5 (512→512, stride=2) → conv6 (512→512, stride=2)

Decoder (upsampling path with skip connections):
    up6→up5→up4→up3→up2→up1→up0

Final: conv_last (64→1) with bilinear upsample to input size.

All conv layers use spectral norm for Lipschitz-constrained training stability.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


class UNetDiscriminatorSN(nn.Module):
    """
    U-Net discriminator with spectral normalization.

    Accepts HR-resolution images (e.g. 256×256) and outputs a per-pixel
    real/fake map of the same spatial size.

    Args:
        in_channels : input image channels (default 3)
        nf          : base number of feature channels (default 64)
        skip_connection : whether to use skip connections in the U-Net decoder
    """

    def __init__(
        self,
        in_channels: int = 3,
        nf: int = 64,
        skip_connection: bool = True,
    ):
        super().__init__()
        self.skip_connection = skip_connection

        # ---- Encoder (downsampling) ----
        # Each encoder block: SN-Conv → LeakyReLU (some with stride=2)
        self.conv0 = spectral_norm(nn.Conv2d(in_channels, nf, 3, 1, 1))

        self.conv1 = spectral_norm(nn.Conv2d(nf, nf * 2, 4, 2, 1))       # /2
        self.conv2 = spectral_norm(nn.Conv2d(nf * 2, nf * 4, 4, 2, 1))   # /4
        self.conv3 = spectral_norm(nn.Conv2d(nf * 4, nf * 8, 4, 2, 1))   # /8

        # Bottleneck convolutions at lowest resolution
        self.conv4 = spectral_norm(nn.Conv2d(nf * 8, nf * 8, 4, 2, 1))   # /16
        self.conv5 = spectral_norm(nn.Conv2d(nf * 8, nf * 8, 4, 2, 1))   # /32
        self.conv6 = spectral_norm(nn.Conv2d(nf * 8, nf * 8, 4, 2, 1))   # /64

        # ---- Decoder (upsampling with skip connections) ----
        # Decoder convolutions: upsample → cat(skip) → SN-Conv
        # Input channels = features + skip (if skip_connection is True)
        skip_ch = nf * 8 if skip_connection else 0

        self.conv7 = spectral_norm(nn.Conv2d(nf * 8 + skip_ch, nf * 8, 3, 1, 1))
        self.conv8 = spectral_norm(nn.Conv2d(nf * 8 + skip_ch, nf * 8, 3, 1, 1))
        self.conv9 = spectral_norm(nn.Conv2d(nf * 8 + skip_ch, nf * 8, 3, 1, 1))

        skip_ch4 = nf * 4 if skip_connection else 0
        self.conv10 = spectral_norm(nn.Conv2d(nf * 8 + skip_ch4, nf * 4, 3, 1, 1))

        skip_ch2 = nf * 2 if skip_connection else 0
        self.conv11 = spectral_norm(nn.Conv2d(nf * 4 + skip_ch2, nf * 2, 3, 1, 1))

        skip_ch1 = nf if skip_connection else 0
        self.conv12 = spectral_norm(nn.Conv2d(nf * 2 + skip_ch1, nf, 3, 1, 1))

        # Final 1×1 conv → single-channel real/fake map
        self.conv_last = spectral_norm(nn.Conv2d(nf, 1, 3, 1, 1))

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W) HR image in [0, 1]
        Returns:
            (B, 1, H, W) per-pixel real/fake logits
        """
        h_in = x.shape[2]
        w_in = x.shape[3]

        # ---- Encoder ----
        e0 = self.lrelu(self.conv0(x))     # (B, 64, H, W)
        e1 = self.lrelu(self.conv1(e0))    # (B, 128, H/2, W/2)
        e2 = self.lrelu(self.conv2(e1))    # (B, 256, H/4, W/4)
        e3 = self.lrelu(self.conv3(e2))    # (B, 512, H/8, W/8)
        e4 = self.lrelu(self.conv4(e3))    # (B, 512, H/16, W/16)
        e5 = self.lrelu(self.conv5(e4))    # (B, 512, H/32, W/32)
        e6 = self.lrelu(self.conv6(e5))    # (B, 512, H/64, W/64)

        # ---- Decoder with skip connections ----
        d6 = F.interpolate(e6, size=e5.shape[2:], mode="bilinear", align_corners=False)
        d6 = self.lrelu(self.conv7(torch.cat([d6, e5], dim=1) if self.skip_connection else d6))

        d5 = F.interpolate(d6, size=e4.shape[2:], mode="bilinear", align_corners=False)
        d5 = self.lrelu(self.conv8(torch.cat([d5, e4], dim=1) if self.skip_connection else d5))

        d4 = F.interpolate(d5, size=e3.shape[2:], mode="bilinear", align_corners=False)
        d4 = self.lrelu(self.conv9(torch.cat([d4, e3], dim=1) if self.skip_connection else d4))

        d3 = F.interpolate(d4, size=e2.shape[2:], mode="bilinear", align_corners=False)
        d3 = self.lrelu(self.conv10(torch.cat([d3, e2], dim=1) if self.skip_connection else d3))

        d2 = F.interpolate(d3, size=e1.shape[2:], mode="bilinear", align_corners=False)
        d2 = self.lrelu(self.conv11(torch.cat([d2, e1], dim=1) if self.skip_connection else d2))

        d1 = F.interpolate(d2, size=e0.shape[2:], mode="bilinear", align_corners=False)
        d1 = self.lrelu(self.conv12(torch.cat([d1, e0], dim=1) if self.skip_connection else d1))

        out = self.conv_last(d1)

        # Bilinear upsample to match exact input spatial size
        # (handles cases where encoder downsampling truncated odd dimensions)
        if out.shape[2] != h_in or out.shape[3] != w_in:
            out = F.interpolate(out, size=(h_in, w_in), mode="bilinear", align_corners=False)

        return out
