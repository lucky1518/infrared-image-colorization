import cv2
import torch
import numpy as np

from skimage.metrics import (
    peak_signal_noise_ratio,
    structural_similarity
)

from generator import Generator

# Device
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# Load Model
model = Generator().to(device)

model.load_state_dict(
    torch.load(
        "models/pix2pix_generator.pth",
        map_location=device
    )
)

model.eval()

# Load Images
inp = cv2.imread(
    "../../data/processed/train/input/input_0.png"
)

gt = cv2.imread(
    "../../data/processed/train/rgb/rgb_0.png"
)

# Convert Ground Truth to RGB and normalize
gt = cv2.cvtColor(
    gt,
    cv2.COLOR_BGR2RGB
) / 255.0

# Prepare Input Tensor
x = inp / 255.0

x = torch.tensor(
    x,
    dtype=torch.float32
).permute(2, 0, 1)

x = x.unsqueeze(0).to(device)

# Generate Prediction
with torch.no_grad():
    pred = model(x)

pred = pred.squeeze().cpu()
pred = pred.permute(1, 2, 0).numpy()

pred = (pred + 1) / 2
pred = np.clip(pred, 0, 1)

# Metrics
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

print("=" * 40)
print("PSNR :", psnr)
print("SSIM :", ssim)
print("=" * 40)
