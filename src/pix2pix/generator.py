
import torch
import torch.nn as nn

class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, normalize=True):
        super().__init__()

        layers = [
            nn.Conv2d(in_size, out_size, 4, 2, 1, bias=False)
        ]

        if normalize:
            layers.append(nn.BatchNorm2d(out_size))

        layers.append(nn.LeakyReLU(0.2))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UNetUp(nn.Module):
    def __init__(self, in_size, out_size):
        super().__init__()

        self.model = nn.Sequential(
            nn.ConvTranspose2d(in_size, out_size, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_size),
            nn.ReLU()
        )

    def forward(self, x, skip):
        x = self.model(x)
        x = torch.cat((x, skip), 1)
        return x


class Generator(nn.Module):

    def __init__(self):
        super().__init__()

        self.down1 = UNetDown(3, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 512, 4, 2, 1),
            nn.ReLU()
        )

        self.up1 = UNetUp(512, 512)
        self.up2 = UNetUp(1024, 256)
        self.up3 = UNetUp(512, 128)
        self.up4 = UNetUp(256, 64)

        self.final = nn.Sequential(
            nn.ConvTranspose2d(128, 3, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, x):

        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)

        b = self.bottleneck(d4)

        u1 = self.up1(b, d4)
        u2 = self.up2(u1, d3)
        u3 = self.up3(u2, d2)
        u4 = self.up4(u3, d1)

        return self.final(u4)
