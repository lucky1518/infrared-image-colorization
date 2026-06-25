import os
import numpy as np
import torch
import matplotlib.pyplot as plt

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

IMAGE_PATH = os.path.join(
    CURRENT_DIR,
    "..",
    "..",
    "data",
    "processed",
    "train_npy",
    "input",
    "input_0.npy"
)

OUTPUT_PATH = os.path.join(
    CURRENT_DIR,
    "pix2pix_output.png"
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
# Load NPY Image
# ---------------------------------

img = np.load(IMAGE_PATH).astype(np.float32)

# Normalize to 0-1
img = img / img.max()

# Convert to tanh range (-1 to 1)
img = img * 2.0 - 1.0

tensor = torch.from_numpy(img).permute(2, 0, 1)
tensor = tensor.unsqueeze(0).to(device)

# ---------------------------------
# Prediction
# ---------------------------------

with torch.no_grad():
    output = model(tensor)

pred = output.squeeze().cpu().permute(1, 2, 0).numpy()

# Convert back from [-1,1] to [0,1]
pred = (pred + 1.0) / 2.0
pred = np.clip(pred, 0, 1)

# ---------------------------------
# Save Output
# ---------------------------------

plt.figure(figsize=(8, 8))
plt.imshow(pred)
plt.axis("off")
plt.savefig(
    OUTPUT_PATH,
    bbox_inches="tight",
    pad_inches=0
)
plt.show()

print(f"Saved: {OUTPUT_PATH}")