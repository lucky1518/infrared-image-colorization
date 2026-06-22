import cv2
import rasterio
import numpy as np
import matplotlib.pyplot as plt

with rasterio.open("data/raw/input/mumbai_input.tif") as src:
    img = src.read()

b5 = img[0]

# Normalize
b5_norm = cv2.normalize(
    b5,
    None,
    0,
    255,
    cv2.NORM_MINMAX
).astype(np.uint8)

# CLAHE
clahe = cv2.createCLAHE(
    clipLimit=2.0,
    tileGridSize=(8,8)
)

enhanced = clahe.apply(b5_norm)

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.imshow(b5_norm, cmap='gray')
plt.title("Original B5")

plt.subplot(1,2,2)
plt.imshow(enhanced, cmap='gray')
plt.title("CLAHE Enhanced")

plt.tight_layout()
plt.show()