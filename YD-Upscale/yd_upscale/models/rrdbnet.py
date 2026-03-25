"""
RRDBNet — Residual-in-Residual Dense Block Network (generator).

Anime 6B configuration: nb=6, nf=64, gc=32, scale=4
Mirrors the architecture of RealESRGAN_x4plus_anime_6B exactly so that
pretrained weights from that model can be loaded directly via the
`params_ema` key.

Architecture
------------
  conv_first(3→64) → 6×RRDB(64,32) → conv_body(64→64) → [long skip]
      → upsample×2 + conv_up1 → upsample×2 + conv_up2
      → conv_hr(64→64) → conv_last(64→3)

Each RRDB contains 3 Residual Dense Blocks (RDB).
Each RDB has 5 dense-connected convolutions with growth channel gc=32.
All residual connections are scaled by β=0.2 for training stability.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualDenseBlock(nn.Module):
    """
    Single Residual Dense Block with 5 dense-connected conv layers.

    Feature maps grow via concatenation at each step:
      x → c1 → cat(x,c1) → c2 → ... → cat(x,c1,c2,c3,c4) → c5
    Output = x + c5 * 0.2  (residual scaling)
    """

    def __init__(self, nf: int = 64, gc: int = 32):
        super().__init__()
        self.conv1 = nn.Conv2d(nf,          gc, 3, 1, 1)
        self.conv2 = nn.Conv2d(nf + gc,     gc, 3, 1, 1)
        self.conv3 = nn.Conv2d(nf + gc * 2, gc, 3, 1, 1)
        self.conv4 = nn.Conv2d(nf + gc * 3, gc, 3, 1, 1)
        self.conv5 = nn.Conv2d(nf + gc * 4, nf, 3, 1, 1)  # back to nf channels
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # Kaiming init — a=0.2 for LeakyReLU slope
        for layer in [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5]:
            nn.init.kaiming_normal_(layer.weight, a=0.2, mode="fan_in")
            nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))  # no activation
        return x + x5 * 0.2


class RRDB(nn.Module):
    """
    Residual-in-Residual Dense Block: 3 chained RDBs with outer residual.
    """

    def __init__(self, nf: int = 64, gc: int = 32):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(nf, gc)
        self.rdb2 = ResidualDenseBlock(nf, gc)
        self.rdb3 = ResidualDenseBlock(nf, gc)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return x + out * 0.2


class RRDBNet(nn.Module):
    """
    Full RRDBNet generator for 4× super-resolution.

    Args:
        in_nc  : input channels  (default 3 = RGB)
        out_nc : output channels (default 3 = RGB)
        nf     : number of feature channels throughout the network (64)
        nb     : number of RRDB blocks (6 for anime-6B, 23 for full)
        gc     : growth channels inside each RDB (32)
        scale  : upscale factor (2 or 4)
    """

    def __init__(
        self,
        in_nc: int = 3,
        out_nc: int = 3,
        nf: int = 64,
        nb: int = 6,
        gc: int = 32,
        scale: int = 4,
    ):
        super().__init__()
        if scale not in (2, 4):
            raise ValueError(f"Only scale 2 or 4 supported, got {scale}")
        self.scale = scale

        # Shallow feature extraction
        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1)

        # Deep feature extraction — N RRDB blocks
        self.body = nn.Sequential(*[RRDB(nf, gc) for _ in range(nb)])
        self.conv_body = nn.Conv2d(nf, nf, 3, 1, 1)

        # Upsampling: nearest-neighbour ×2 + conv, repeated for scale=4
        self.conv_up1 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(nf, nf, 3, 1, 1) if scale == 4 else None

        # HR refinement + final projection
        self.conv_hr   = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.conv_first(x)                       # shallow features
        body_feat = self.conv_body(self.body(feat))      # deep features
        feat = feat + body_feat                          # long skip connection

        # Upsample ×2
        feat = self.lrelu(self.conv_up1(
            F.interpolate(feat, scale_factor=2, mode="nearest")
        ))
        # Upsample ×2 again for scale=4
        if self.scale == 4:
            feat = self.lrelu(self.conv_up2(
                F.interpolate(feat, scale_factor=2, mode="nearest")
            ))

        out = self.lrelu(self.conv_hr(feat))
        out = self.conv_last(out)
        return out
