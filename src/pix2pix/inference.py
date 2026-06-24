import os
import cv2
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
    "train",
    "input",
    "input_0.png"
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
# Load Image
# ---------------------------------

img = cv2.imread(IMAGE_PATH)

if img is None:
    raise FileNotFoundError(
        f"Image not found: {IMAGE_PATH}"
    )

img = img / 255.0

tensor = torch.tensor(
    img,
    dtype=torch.float32
).permute(2, 0, 1)

tensor = tensor.unsqueeze(0).to(device)

# ---------------------------------
# Prediction
# ---------------------------------

with torch.no_grad():
    output = model(tensor)

pred = output.squeeze().cpu()

pred = pred.permute(1, 2, 0).numpy()

pred = (pred + 1) / 2

# ---------------------------------
# Save Output
# ---------------------------------

plt.figure(figsize=(8, 8))
plt.imshow(pred)
plt.axis("off")
plt.savefig(OUTPUT_PATH)
plt.show()

print(f"Saved: {OUTPUT_PATH}")