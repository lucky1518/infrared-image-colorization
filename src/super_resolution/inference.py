import os
import time
import numpy as np
import torch

from model import EDSR

# ==========================================
# Configuration
# ==========================================

INPUT_FILE = "data/processed/train_sr/input/input_0.npy"

MODEL_PATH = "models/edsr_x2_best.pth"

OUTPUT_DIR = "results"

OUTPUT_FILE = "sr_output.npy"

# ==========================================
# Device
# ==========================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 50)
print("Device :", device)
print("=" * 50)

# ==========================================
# Create Output Folder
# ==========================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

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
# Load Image
# ==========================================

lr = np.load(INPUT_FILE).astype(np.float32)

lr = torch.from_numpy(lr)

lr = lr.permute(2, 0, 1)

lr = lr.unsqueeze(0).to(device)

print("Input Shape :", lr.shape)

# ==========================================
# Inference
# ==========================================

start = time.time()

with torch.no_grad():

    sr = model(lr)

elapsed = time.time() - start

print("Output Shape:", sr.shape)

print(f"Inference Time : {elapsed:.4f} sec")

# ==========================================
# Save Output
# ==========================================

sr = sr.squeeze(0)

sr = sr.permute(1, 2, 0)

sr = sr.cpu().numpy()

np.save(
    os.path.join(
        OUTPUT_DIR,
        OUTPUT_FILE
    ),
    sr
)

print("Output Saved Successfully.")