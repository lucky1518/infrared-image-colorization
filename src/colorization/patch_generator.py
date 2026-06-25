import os
import cv2
import rasterio
import numpy as np

PATCH_SIZE = 256

# ==========================
# Input Directories
# ==========================

INPUT_DIR = "data/raw/input"
RGB_DIR = "data/raw/rgb"

# ==========================
# PNG Output (Visualization)
# ==========================

OUTPUT_INPUT_PNG = "data/processed/train/input"
OUTPUT_RGB_PNG = "data/processed/train/rgb"

# ==========================
# NPY Output (Training)
# ==========================

OUTPUT_INPUT_NPY = "data/processed/train_npy/input"
OUTPUT_RGB_NPY = "data/processed/train_npy/rgb"

# ==========================
# Create Folders
# ==========================

os.makedirs(OUTPUT_INPUT_PNG, exist_ok=True)
os.makedirs(OUTPUT_RGB_PNG, exist_ok=True)

os.makedirs(OUTPUT_INPUT_NPY, exist_ok=True)
os.makedirs(OUTPUT_RGB_NPY, exist_ok=True)

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

    # --------------------------
    # Read TIFF Images
    # --------------------------

    with rasterio.open(input_path) as src:
        input_img = src.read()

    with rasterio.open(rgb_path) as src:
        rgb_img = src.read()

    # (Bands, H, W) -> (H, W, Bands)

    input_img = np.transpose(input_img, (1, 2, 0))
    rgb_img = np.transpose(rgb_img, (1, 2, 0))

    height, width, _ = input_img.shape

    for y in range(0, height - PATCH_SIZE, PATCH_SIZE):
        for x in range(0, width - PATCH_SIZE, PATCH_SIZE):

            # --------------------------
            # Original Patches
            # --------------------------

            input_patch = input_img[
                y:y + PATCH_SIZE,
                x:x + PATCH_SIZE
            ]

            rgb_patch = rgb_img[
                y:y + PATCH_SIZE,
                x:x + PATCH_SIZE
            ]

            # ===================================================
            # SAVE ORIGINAL PATCHES (.NPY)
            # ===================================================

            np.save(
                os.path.join(
                    OUTPUT_INPUT_NPY,
                    f"input_{count}.npy"
                ),
                input_patch.astype(np.float32)
            )

            np.save(
                os.path.join(
                    OUTPUT_RGB_NPY,
                    f"rgb_{count}.npy"
                ),
                rgb_patch.astype(np.float32)
            )

            # ===================================================
            # CREATE PNG ONLY FOR VISUALIZATION
            # ===================================================

            input_png = cv2.normalize(
                input_patch,
                None,
                0,
                255,
                cv2.NORM_MINMAX
            ).astype(np.uint8)

            rgb_png = cv2.normalize(
                rgb_patch,
                None,
                0,
                255,
                cv2.NORM_MINMAX
            ).astype(np.uint8)

            cv2.imwrite(
                os.path.join(
                    OUTPUT_INPUT_PNG,
                    f"input_{count}.png"
                ),
                input_png
            )

            cv2.imwrite(
                os.path.join(
                    OUTPUT_RGB_PNG,
                    f"rgb_{count}.png"
                ),
                rgb_png
            )

            count += 1

print("\n======================================")
print("TOTAL PATCHES GENERATED :", count)
print("PNG Dataset Saved       :", OUTPUT_INPUT_PNG)
print("NPY Dataset Saved       :", OUTPUT_INPUT_NPY)
print("======================================")