import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import LandsatDataset
from model import UNet

# Create models folder if missing
os.makedirs("models", exist_ok=True)

# Dataset
dataset = LandsatDataset(
    "data/processed/train/input",
    "data/processed/train/rgb"
)

print("=" * 50)
print("Dataset Size:", len(dataset))
print("=" * 50)

# Data Loader
loader = DataLoader(
    dataset,
    batch_size=4,      # Increased from 2
    shuffle=True
)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using Device:", device)

# Model
model = UNet().to(device)

# Loss Function
criterion = nn.MSELoss()

# Optimizer
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

# Training Parameters
EPOCHS = 20

print("\nTraining Started...\n")

for epoch in range(EPOCHS):

    running_loss = 0.0

    for inputs, targets in loader:

        inputs = inputs.to(device)
        targets = targets.to(device)

        outputs = model(inputs)

        loss = criterion(outputs, targets)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(loader)

    print(
        f"Epoch [{epoch+1}/{EPOCHS}] "
        f"Average Loss: {avg_loss:.6f}"
    )

# Save Model
torch.save(
    model.state_dict(),
    "models/unet.pth"
)

print("\n=================================")
print("Training Completed Successfully!")
print("Model Saved: models/unet.pth")
print("=================================")