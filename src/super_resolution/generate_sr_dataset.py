import os
import cv2
import numpy as np

# ==========================================
# Configuration
# ==========================================

SCALE_FACTOR = 2

INPUT_DIR = "data/processed/train_npy/input"

OUTPUT_INPUT = "data/processed/train_sr/input"
OUTPUT_TARGET = "data/processed/train_sr/target"

os.makedirs(OUTPUT_INPUT, exist_ok=True)
os.makedirs(OUTPUT_TARGET, exist_ok=True)

# ==========================================
# Generate Super-Resolution Dataset
# ==========================================

files = sorted(os.listdir(INPUT_DIR))

count = 0

print("=" * 50)
print("Generating Super-Resolution Dataset")
print("=" * 50)

for file in files:

    if not file.endswith(".npy"):
        continue

    # ---------------------------------
    # Load High Resolution Image (256x256)
    # ---------------------------------

    hr = np.load(
        os.path.join(INPUT_DIR, file)
    ).astype(np.float32)

    h, w, c = hr.shape

    # ---------------------------------
    # Create Low Resolution Image (128x128)
    # ---------------------------------

    lr = cv2.resize(
        hr,
        (w // SCALE_FACTOR, h // SCALE_FACTOR),
        interpolation=cv2.INTER_AREA
    )

    # ---------------------------------
    # Save Low Resolution Input
    # Shape : (128,128,3)
    # ---------------------------------

    np.save(
        os.path.join(OUTPUT_INPUT, file),
        lr.astype(np.float32)
    )

    # ---------------------------------
    # Save High Resolution Target
    # Shape : (256,256,3)
    # ---------------------------------

    np.save(
        os.path.join(OUTPUT_TARGET, file),
        hr.astype(np.float32)
    )

    count += 1

print("\n" + "=" * 50)
print("Dataset Generation Complete")
print("Total Images :", count)
print("=" * 50)