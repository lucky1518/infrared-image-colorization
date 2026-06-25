import os
import time
import numpy as np
import torch

from skimage.metrics import (
    peak_signal_noise_ratio,
    structural_similarity
)

from model import EDSR

# ==========================================
# Configuration
# ==========================================

INPUT_FILE = "data/processed/train_sr/input/input_0.npy"

TARGET_FILE = "data/processed/train_sr/target/input_0.npy"

MODEL_PATH = "models/edsr_x2_best.pth"

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
# Load Model
# ==========================================

model = EDSR().to(device)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model.eval()

print("Model Loaded Successfully.")

# ==========================================
# Load Images
# ==========================================

lr = np.load(INPUT_FILE).astype(np.float32)
hr = np.load(TARGET_FILE).astype(np.float32)

lr_tensor = torch.from_numpy(lr)
lr_tensor = lr_tensor.permute(2, 0, 1)
lr_tensor = lr_tensor.unsqueeze(0).to(device)

# ==========================================
# Inference
# ==========================================

start = time.time()

with torch.no_grad():
    sr = model(lr_tensor)

elapsed = time.time() - start

# ==========================================
# Convert Prediction
# ==========================================

sr = sr.squeeze(0)
sr = sr.permute(1, 2, 0)
sr = sr.cpu().numpy()

sr = np.clip(sr, 0.0, 1.0)
hr = np.clip(hr, 0.0, 1.0)

# ==========================================
# Metrics
# ==========================================

psnr = peak_signal_noise_ratio(
    hr,
    sr,
    data_range=1.0
)

ssim = structural_similarity(
    hr,
    sr,
    channel_axis=2,
    data_range=1.0
)

# ==========================================
# Results
# ==========================================

print("\n" + "=" * 60)
print("Evaluation Results")
print("=" * 60)

print(f"PSNR           : {psnr:.4f} dB")
print(f"SSIM           : {ssim:.4f}")
print(f"Inference Time : {elapsed:.4f} sec")

print("=" * 60)