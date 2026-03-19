import torch
import torch.nn as nn
import torch.nn.functional as F


class SRVGGNetCompact(nn.Module):
    def __init__(
        self,
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_conv=32,
        upscale=4,
        act_type="prelu",
    ):
        super().__init__()
        self.upscale = upscale

        if act_type == "relu":
            def make_act():
                return nn.ReLU(inplace=True)
        elif act_type == "prelu":
            def make_act():
                return nn.PReLU(num_parameters=num_feat)
        elif act_type == "leakyrelu":
            def make_act():
                return nn.LeakyReLU(negative_slope=0.1, inplace=True)
        else:
            raise ValueError(f"Unsupported act_type: {act_type}")

        layers = [
            nn.Conv2d(num_in_ch, num_feat, 3, 1, 1),
            make_act(),
        ]

        for _ in range(num_conv):
            layers.extend([
                nn.Conv2d(num_feat, num_feat, 3, 1, 1),
                make_act(),
            ])

        layers.append(nn.Conv2d(num_feat, num_out_ch * upscale * upscale, 3, 1, 1))

        self.body = nn.Sequential(*layers)
        self.upsampler = nn.PixelShuffle(upscale)

    def forward(self, x):
        residual = self.body(x)
        residual = self.upsampler(residual)

        # IMPORTANT: bicubic skip instead of nearest
        base = F.interpolate(
            x,
            scale_factor=self.upscale,
            mode="bicubic",
            align_corners=False,
        )

        return residual + base