import os
import numpy as np

INPUT_DIR = "data/processed/train_sr/input"
TARGET_DIR = "data/processed/train_sr/target"

input_files = sorted(os.listdir(INPUT_DIR))
target_files = sorted(os.listdir(TARGET_DIR))

print("=" * 50)
print("Super Resolution Dataset Check")
print("=" * 50)

print("Input Files :", len(input_files))
print("Target Files:", len(target_files))

sample = input_files[0]

lr = np.load(os.path.join(INPUT_DIR, sample))
hr = np.load(os.path.join(TARGET_DIR, sample))

print("\nSample :", sample)

print("\nLow Resolution")
print("Shape :", lr.shape)
print("Min   :", lr.min())
print("Max   :", lr.max())

print("\nHigh Resolution")
print("Shape :", hr.shape)
print("Min   :", hr.min())
print("Max   :", hr.max())

print("\nDataset Verified Successfully!")
print("=" * 50)