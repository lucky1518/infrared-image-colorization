
import cv2
import torch
import matplotlib.pyplot as plt

from generator import Generator

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# Load model
model = Generator().to(device)

model.load_state_dict(
    torch.load(
        "models/pix2pix_generator.pth",
        map_location=device
    )
)

model.eval()

# Load images
inp = cv2.imread(
    "data/processed/train/input/input_318.png"
)

gt = cv2.imread(
    "data/processed/train/rgb/rgb_318.png"
)

# Prepare tensor
x = inp / 255.0

x = torch.tensor(
    x,
    dtype=torch.float32
).permute(2,0,1)

x = x.unsqueeze(0).to(device)

# Prediction
with torch.no_grad():
    pred = model(x)

pred = pred.squeeze().cpu()
pred = pred.permute(1,2,0).numpy()

pred = (pred + 1) / 2
pred = pred.clip(0,1)

# Display
plt.figure(figsize=(18,6))

plt.subplot(1,3,1)
plt.imshow(cv2.cvtColor(inp, cv2.COLOR_BGR2RGB))
plt.title("Input IR")
plt.axis("off")

plt.subplot(1,3,2)
plt.imshow(cv2.cvtColor(gt, cv2.COLOR_BGR2RGB))
plt.title("Ground Truth RGB")
plt.axis("off")

plt.subplot(1,3,3)
plt.imshow(pred)
plt.title("Pix2Pix Output")
plt.axis("off")

plt.tight_layout()
plt.savefig("comparison.png")
plt.show()

print("Saved: comparison.png")
