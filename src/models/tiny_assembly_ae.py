
import torch
import torch.nn as nn


# ==========================================
# DEPTHWISE SEPARABLE CONVOLUTION
# ==========================================

class DSConv(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.block = nn.Sequential(

            # Depthwise convolution
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=in_channels,
                bias=False
            ),

            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True),

            # Pointwise convolution
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                bias=False
            ),

            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


# ==========================================
# TINY ASSEMBLY AUTOENCODER
# ==========================================

class TinyAssemblyAE(nn.Module):

    def __init__(self):
        super().__init__()

        # ENCODER
        self.encoder = nn.Sequential(

            nn.Conv2d(
                3, 16,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(16),
            nn.SiLU(inplace=True),

            DSConv(16, 32, stride=2),
            DSConv(32, 64, stride=2),
            DSConv(64, 128, stride=2)
        )

        # BOTTLENECK
        self.bottleneck = nn.Sequential(

            nn.Conv2d(128, 32, kernel_size=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True),

            nn.Conv2d(32, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.SiLU(inplace=True)
        )

        # DECODER
        self.decoder = nn.Sequential(

            nn.Upsample(
                scale_factor=2,
                mode="bilinear",
                align_corners=False
            ),

            DSConv(128, 64),

            nn.Upsample(
                scale_factor=2,
                mode="bilinear",
                align_corners=False
            ),

            DSConv(64, 32),

            nn.Upsample(
                scale_factor=2,
                mode="bilinear",
                align_corners=False
            ),

            DSConv(32, 16),

            nn.Upsample(
                scale_factor=2,
                mode="bilinear",
                align_corners=False
            ),

            nn.Conv2d(16, 3, kernel_size=3, padding=1),

            nn.Sigmoid()
        )

    def forward(self, x):

        encoded = self.encoder(x)

        compressed = self.bottleneck(encoded)

        reconstructed = self.decoder(compressed)

        return reconstructed


# ==========================================
# MODEL PARAMETERS
# ==========================================

def count_parameters(model):

    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )
