import rasterio

input_path = "data/raw/input/mumbai_input.tif"
rgb_path = "data/raw/rgb/mumbai_rgb.tif"

print("=" * 50)
print("READING INPUT IMAGE")
print("=" * 50)

with rasterio.open(input_path) as src:
    input_img = src.read()

print("Input Shape:", input_img.shape)

print()

print("=" * 50)
print("READING RGB IMAGE")
print("=" * 50)

with rasterio.open(rgb_path) as src:
    rgb_img = src.read()

print("RGB Shape:", rgb_img.shape)