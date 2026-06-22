import cv2
import rasterio
import matplotlib.pyplot as plt
import numpy as np

with rasterio.open("data/raw/input/mumbai_input.tif") as src:
    img = src.read()

b5 = img[0]

# Normalize
b5 = cv2.normalize(
    b5,
    None,
    0,
    255,
    cv2.NORM_MINMAX
).astype(np.uint8)

# Denoise
denoised = cv2.fastNlMeansDenoising(
    b5,
    None,
    h=4,
    templateWindowSize=7,
    searchWindowSize=21
)

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.imshow(b5, cmap="gray")
plt.title("Original")

plt.subplot(1,2,2)
plt.imshow(denoised, cmap="gray")
plt.title("Denoised")

plt.tight_layout()
plt.show()