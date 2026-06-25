import torch

from model import EDSR

# ==========================================
# Create Model
# ==========================================

model = EDSR()

print("=" * 50)
print("EDSR Model Created Successfully")
print("=" * 50)

# ==========================================
# Dummy Input
# ==========================================

x = torch.randn(
    1,      # Batch Size
    3,      # RGB Channels
    128,    # Height
    128     # Width
)

print("Input Shape :", x.shape)

# ==========================================
# Forward Pass
# ==========================================

with torch.no_grad():
    y = model(x)

print("Output Shape:", y.shape)

print("\n" + "=" * 50)
print("Model Test Passed Successfully!")
print("=" * 50)