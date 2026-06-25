import torch
import torch.nn as nn
# ==========================================
# Configuration
# ==========================================

N_RESBLOCKS = 16
N_FEATURES = 64
UPSCALE_FACTOR = 2

# ==========================================
# Residual Block
# ==========================================

class ResidualBlock(nn.Module):

    def __init__(self, channels):
        super().__init__()

        self.conv1 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1
        )

        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1
        )

    def forward(self, x):

        identity = x

        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)

        out = out + identity

        return out
# ==========================================
# Upsampler
# ==========================================

class Upsampler(nn.Sequential):

    def __init__(self, scale, n_features):

        layers = []

        if scale == 2:

            layers.append(
                nn.Conv2d(
                    n_features,
                    n_features * 4,
                    kernel_size=3,
                    padding=1
                )
            )

            layers.append(
                nn.PixelShuffle(2)
            )

        else:
            raise NotImplementedError(
                "Only x2 Super Resolution is supported."
            )

        super().__init__(*layers)

# ==========================================
# EDSR Baseline
# ==========================================

class EDSR(nn.Module):

    def __init__(self):
        super().__init__()

        # Head
        self.head = nn.Conv2d(
            3,
            N_FEATURES,
            kernel_size=3,
            padding=1
        )

        # Body (16 Residual Blocks)
        self.body = nn.Sequential(
            *[
                ResidualBlock(N_FEATURES)
                for _ in range(N_RESBLOCKS)
            ]
        )

        self.body_conv = nn.Conv2d(
            N_FEATURES,
            N_FEATURES,
            kernel_size=3,
            padding=1
        )

        # Upsampling
        self.upsampler = Upsampler(
            UPSCALE_FACTOR,
            N_FEATURES
        )

        # Tail
        self.tail = nn.Conv2d(
            N_FEATURES,
            3,
            kernel_size=3,
            padding=1
        )

    def forward(self, x):

        # Head
        x = self.head(x)

        # Global Skip Connection
        residual = x

        # Body
        x = self.body(x)
        x = self.body_conv(x)

        x = x + residual

        # Upsampling
        x = self.upsampler(x)

        # Tail
        x = self.tail(x)

        return x