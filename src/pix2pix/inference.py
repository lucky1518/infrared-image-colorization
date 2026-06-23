
import cv2
import torch
import matplotlib.pyplot as plt

from generator import Generator

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

# Load Image
img = cv2.imread(
    "../../data/processed/train/input/input_0.png"
)

img = img / 255.0

tensor = torch.tensor(
    img,
    dtype=torch.float32
).permute(2,0,1)

tensor = tensor.unsqueeze(0).to(device)

# Prediction
with torch.no_grad():

    output = model(tensor)

pred = output.squeeze().cpu()

pred = pred.permute(1,2,0).numpy()

pred = (pred + 1) / 2

plt.figure(figsize=(8,8))
plt.imshow(pred)
plt.axis("off")
plt.savefig("pix2pix_output.png")
plt.show()

print("Saved: pix2pix_output.png")
