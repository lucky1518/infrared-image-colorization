import os
import cv2
import torch
import numpy as np

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
    "train",
    "input",
    "input_0.png"
)

GROUND_TRUTH = os.path.join(
    CURRENT_DIR,
    "..",
    "..",
    "data",
    "processed",
    "train",
    "rgb",
    "rgb_0.png"
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
# Load Images
# ---------------------------------

inp = cv2.imread(INPUT_IMAGE)

gt = cv2.imread(GROUND_TRUTH)

if inp is None:
    raise FileNotFoundError(
        f"Input image not found: {INPUT_IMAGE}"
    )

if gt is None:
    raise FileNotFoundError(
        f"Ground truth image not found: {GROUND_TRUTH}"
    )

gt = cv2.cvtColor(
    gt,
    cv2.COLOR_BGR2RGB
) / 255.0

# ---------------------------------
# Prepare Input
# ---------------------------------

x = inp / 255.0

x = torch.tensor(
    x,
    dtype=torch.float32
).permute(2, 0, 1)

x = x.unsqueeze(0).to(device)

# ---------------------------------
# Generate Prediction
# ---------------------------------

with torch.no_grad():
    pred = model(x)

pred = pred.squeeze().cpu()

pred = pred.permute(1, 2, 0).numpy()

pred = (pred + 1) / 2

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