import rasterio
import matplotlib.pyplot as plt
import numpy as np

# Read Input Image
with rasterio.open("data/raw/input/mumbai_input.tif") as src:
    input_img = src.read()

# Read RGB Image
with rasterio.open("data/raw/rgb/mumbai_rgb.tif") as src:
    rgb_img = src.read()

# Extract Bands
b5 = input_img[0]
b6 = input_img[1]
b7 = input_img[2]

# RGB Image
rgb = np.dstack([
    rgb_img[0],
    rgb_img[1],
    rgb_img[2]
])

# Normalize RGB
rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min())

plt.figure(figsize=(15,8))

plt.subplot(2,2,1)
plt.imshow(b5, cmap='gray')
plt.title("B5 - NIR")

plt.subplot(2,2,2)
plt.imshow(b6, cmap='gray')
plt.title("B6 - SWIR1")

plt.subplot(2,2,3)
plt.imshow(b7, cmap='gray')
plt.title("B7 - SWIR2")

plt.subplot(2,2,4)
plt.imshow(rgb)
plt.title("RGB Ground Truth")

plt.tight_layout()
plt.show()