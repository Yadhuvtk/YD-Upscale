import torch
import torch.nn as nn
import torch.nn.functional as F


class SRVGGNetCompact(nn.Module):
    def __init__(
        self,
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_conv=16,
        upscale=4,
        act_type="prelu",
    ):
        super().__init__()
        self.upscale = upscale

        if act_type == "relu":
            activation = nn.ReLU(inplace=True)
        elif act_type == "prelu":
            activation = nn.PReLU(num_parameters=num_feat)
        elif act_type == "leakyrelu":
            activation = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        else:
            raise ValueError(f"Unsupported act_type: {act_type}")

        body = [nn.Conv2d(num_in_ch, num_feat, 3, 1, 1), activation]

        for _ in range(num_conv):
            if act_type == "prelu":
                act = nn.PReLU(num_parameters=num_feat)
            elif act_type == "relu":
                act = nn.ReLU(inplace=True)
            else:
                act = nn.LeakyReLU(negative_slope=0.1, inplace=True)

            body.extend([
                nn.Conv2d(num_feat, num_feat, 3, 1, 1),
                act
            ])

        body.append(nn.Conv2d(num_feat, num_out_ch * upscale * upscale, 3, 1, 1))
        self.body = nn.Sequential(*body)
        self.upsampler = nn.PixelShuffle(upscale)

    def forward(self, x):
        out = self.body(x)
        out = self.upsampler(out)
        base = F.interpolate(x, scale_factor=self.upscale, mode="nearest")
        return out + base