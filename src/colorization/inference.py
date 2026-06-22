import os
import torch
import cv2
import matplotlib.pyplot as plt

from model import UNet

# Device
device = torch.device("cpu")

# Load Model
model = UNet().to(device)

model.load_state_dict(
    torch.load(
        "models/unet.pth",
        map_location=device
    )
)

model.eval()

# Load Images
img = cv2.imread(
    "data/processed/train/input/input_0.png"
)

img_rgb = cv2.imread(
    "data/processed/train/rgb/rgb_0.png"
)

# Normalize
x = img / 255.0

x = torch.tensor(
    x,
    dtype=torch.float32
).permute(2, 0, 1)

x = x.unsqueeze(0)

# Prediction
with torch.no_grad():
    pred = model(x)

pred = pred.squeeze(0)

pred = pred.permute(
    1, 2, 0
).numpy()

# Save Prediction
os.makedirs("outputs", exist_ok=True)

pred_uint8 = (pred * 255).astype("uint8")

cv2.imwrite(
    "outputs/prediction.png",
    cv2.cvtColor(
        pred_uint8,
        cv2.COLOR_RGB2BGR
    )
)

print("Prediction Saved!")

# Plot
plt.figure(figsize=(15,5))

plt.subplot(1,3,1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("Input")

plt.subplot(1,3,2)
plt.imshow(cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB))
plt.title("Ground Truth")

plt.subplot(1,3,3)
plt.imshow(pred)
plt.title("Prediction")

plt.tight_layout()
plt.show()