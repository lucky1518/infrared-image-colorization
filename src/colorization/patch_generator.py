import os
import cv2
import rasterio
import numpy as np

PATCH_SIZE = 256

INPUT_DIR = "data/raw/input"
RGB_DIR = "data/raw/rgb"

OUTPUT_INPUT = "data/processed/train/input"
OUTPUT_RGB = "data/processed/train/rgb"

os.makedirs(OUTPUT_INPUT, exist_ok=True)
os.makedirs(OUTPUT_RGB, exist_ok=True)

count = 0

input_files = sorted(os.listdir(INPUT_DIR))

for input_file in input_files:

    if not input_file.endswith(".tif"):
        continue

    city_name = input_file.replace("_input.tif", "")

    rgb_file = city_name + "_rgb.tif"

    input_path = os.path.join(INPUT_DIR, input_file)
    rgb_path = os.path.join(RGB_DIR, rgb_file)

    if not os.path.exists(rgb_path):
        print(f"Missing RGB file for {city_name}")
        continue

    print(f"\nProcessing {city_name}")

    with rasterio.open(input_path) as src:
        input_img = src.read()

    with rasterio.open(rgb_path) as src:
        rgb_img = src.read()

    input_img = np.transpose(input_img, (1, 2, 0))
    rgb_img = np.transpose(rgb_img, (1, 2, 0))

    height, width, _ = input_img.shape

    for y in range(0, height - PATCH_SIZE, PATCH_SIZE):
        for x in range(0, width - PATCH_SIZE, PATCH_SIZE):

            input_patch = input_img[
                y:y+PATCH_SIZE,
                x:x+PATCH_SIZE
            ]

            rgb_patch = rgb_img[
                y:y+PATCH_SIZE,
                x:x+PATCH_SIZE
            ]

            input_patch = cv2.normalize(
                input_patch,
                None,
                0,
                255,
                cv2.NORM_MINMAX
            ).astype(np.uint8)

            rgb_patch = cv2.normalize(
                rgb_patch,
                None,
                0,
                255,
                cv2.NORM_MINMAX
            ).astype(np.uint8)

            cv2.imwrite(
                os.path.join(
                    OUTPUT_INPUT,
                    f"input_{count}.png"
                ),
                input_patch
            )

            cv2.imwrite(
                os.path.join(
                    OUTPUT_RGB,
                    f"rgb_{count}.png"
                ),
                rgb_patch
            )

            count += 1

print("\n=================================")
print("TOTAL PATCHES GENERATED:", count)
print("=================================")