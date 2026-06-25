import os
import time
import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from dataset import SRDataset
from model import EDSR

# ==========================================
# Configuration
# ==========================================

INPUT_DIR = "data/processed/train_sr/input"
TARGET_DIR = "data/processed/train_sr/target"

MODEL_DIR = "models"
MODEL_NAME = "edsr_x2_best.pth"

BATCH_SIZE = 8
EPOCHS = 20
LEARNING_RATE = 1e-4

# ==========================================
# Device
# ==========================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("Device :", device)
print("=" * 60)

# ==========================================
# Dataset
# ==========================================

dataset = SRDataset(
    INPUT_DIR,
    TARGET_DIR
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

print("Dataset Size :", len(dataset))

# ==========================================
# Model
# ==========================================

model = EDSR().to(device)

# ==========================================
# Loss
# ==========================================

criterion = nn.L1Loss()

# ==========================================
# Optimizer
# ==========================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

# ==========================================
# Scheduler
# ==========================================

scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=10,
    gamma=0.5
)

# ==========================================
# Model Folder
# ==========================================

os.makedirs(MODEL_DIR, exist_ok=True)

best_loss = float("inf")

start_time = time.time()

print("\nTraining Started...\n")

# ==========================================
# Training Loop
# ==========================================

for epoch in range(EPOCHS):

    model.train()

    running_loss = 0.0

    for lr, hr in loader:

        lr = lr.to(device)
        hr = hr.to(device)

        optimizer.zero_grad()

        sr = model(lr)

        loss = criterion(sr, hr)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    scheduler.step()

    epoch_loss = running_loss / len(loader)

    print(
        f"Epoch {epoch+1:02d}/{EPOCHS} | "
        f"Loss: {epoch_loss:.6f} | "
        f"LR: {scheduler.get_last_lr()[0]:.6f}"
    )

    if epoch_loss < best_loss:

        best_loss = epoch_loss

        torch.save(
            model.state_dict(),
            os.path.join(MODEL_DIR, MODEL_NAME)
        )

        print("Best model updated.")

# ==========================================
# Training Summary
# ==========================================

elapsed = time.time() - start_time

print("\n" + "=" * 60)
print("Training Completed")
print("Best Loss :", round(best_loss, 6))
print("Time Taken:", round(elapsed / 60, 2), "minutes")
print("Model Saved:", os.path.join(MODEL_DIR, MODEL_NAME))
print("=" * 60)