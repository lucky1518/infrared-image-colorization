import os
import numpy as np
import torch

from skimage.metrics import (
    peak_signal_noise_ratio,
    structural_similarity
)

from generator import Generator

# ---------------------------------
# Paths
# ---------------------------------

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    CURRENT_DIR,
    "..",
    "..",
    "models",
    "pix2pix_generator.pth"
)

INPUT_IMAGE = os.path.join(
    CURRENT_DIR,
    "..",
    "..",
    "data",
    "processed",
    "train_npy",
    "input",
    "input_0.npy"
)

GROUND_TRUTH = os.path.join(
    CURRENT_DIR,
    "..",
    "..",
    "data",
    "processed",
    "train_npy",
    "rgb",
    "rgb_0.npy"
)

# ---------------------------------
# Device
# ---------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using Device:", device)

# ---------------------------------
# Load Model
# ---------------------------------

model = Generator().to(device)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model.eval()

print("Model Loaded")

# ---------------------------------
# Load NPY Images
# ---------------------------------

inp = np.load(INPUT_IMAGE).astype(np.float32)
gt = np.load(GROUND_TRUTH).astype(np.float32)

# Normalize to 0-1
inp = inp / inp.max()
gt = gt / gt.max()

# Convert input to tanh range
inp = inp * 2.0 - 1.0

# ---------------------------------
# Prepare Input
# ---------------------------------

x = torch.from_numpy(inp).permute(2, 0, 1)
x = x.unsqueeze(0).to(device)

# ---------------------------------
# Prediction
# ---------------------------------

with torch.no_grad():
    pred = model(x)

pred = pred.squeeze().cpu().permute(1, 2, 0).numpy()

# Convert back from tanh
pred = (pred + 1.0) / 2.0
pred = np.clip(pred, 0, 1)

# ---------------------------------
# Metrics
# ---------------------------------

psnr = peak_signal_noise_ratio(
    gt,
    pred,
    data_range=1.0
)

ssim = structural_similarity(
    gt,
    pred,
    channel_axis=2,
    data_range=1.0
)

print("\n" + "=" * 40)
print("PSNR :", round(psnr, 4))
print("SSIM :", round(ssim, 4))
print("=" * 40)