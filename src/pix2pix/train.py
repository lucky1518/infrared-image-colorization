
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from generator import Generator
from discriminator import Discriminator

import sys
sys.path.append("../colorization")

from dataset import LandsatDataset

# -------------------
# Folders
# -------------------

os.makedirs("models", exist_ok=True)

# -------------------
# Dataset
# -------------------

dataset = LandsatDataset(
    "../../data/processed/train/input",
    "../../data/processed/train/rgb"
)

print("=" * 50)
print("Dataset Size:", len(dataset))
print("=" * 50)

loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True
)

# -------------------
# Device
# -------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using Device:", device)

# -------------------
# Models
# -------------------

G = Generator().to(device)
D = Discriminator().to(device)

# -------------------
# Losses
# -------------------

gan_loss = nn.BCEWithLogitsLoss()
l1_loss = nn.L1Loss()

# -------------------
# Optimizers
# -------------------

opt_G = torch.optim.Adam(
    G.parameters(),
    lr=0.0002,
    betas=(0.5, 0.999)
)

opt_D = torch.optim.Adam(
    D.parameters(),
    lr=0.0002,
    betas=(0.5, 0.999)
)

# -------------------
# Training
# -------------------

EPOCHS = 20

print("\nPix2Pix Training Started...\n")

for epoch in range(EPOCHS):

    for inputs, targets in loader:

        inputs = inputs.to(device)
        targets = targets.to(device)

        # -------------------
        # Train Generator
        # -------------------

        fake = G(inputs)

        pred_fake = D(inputs, fake)

        valid = torch.ones_like(pred_fake)

        g_gan = gan_loss(
            pred_fake,
            valid
        )

        g_l1 = l1_loss(
            fake,
            targets
        )

        g_loss = g_gan + 100 * g_l1

        opt_G.zero_grad()
        g_loss.backward()
        opt_G.step()

        # -------------------
        # Train Discriminator
        # -------------------

        pred_real = D(
            inputs,
            targets
        )

        pred_fake = D(
            inputs,
            fake.detach()
        )

        valid = torch.ones_like(pred_real)
        fake_label = torch.zeros_like(pred_fake)

        d_real = gan_loss(
            pred_real,
            valid
        )

        d_fake = gan_loss(
            pred_fake,
            fake_label
        )

        d_loss = (
            d_real + d_fake
        ) / 2

        opt_D.zero_grad()
        d_loss.backward()
        opt_D.step()

    print(
        f"Epoch [{epoch+1}/{EPOCHS}] "
        f"G Loss: {g_loss.item():.4f} "
        f"D Loss: {d_loss.item():.4f}"
    )

# -------------------
# Save Model
# -------------------

torch.save(
    G.state_dict(),
    "models/pix2pix_generator.pth"
)

print("\n=================================")
print("Pix2Pix Training Completed!")
print("Model Saved")
print("=================================")
