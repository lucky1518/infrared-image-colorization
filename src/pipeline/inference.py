import os
import sys
import numpy as np
import torch

# ==========================================
# Project Root
# ==========================================

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

PROJECT_ROOT = os.path.abspath(
    os.path.join(CURRENT_DIR, "..", "..")
)

sys.path.append(PROJECT_ROOT)

# ==========================================
# Import Models
# ==========================================

from src.super_resolution.model import EDSR
from src.pix2pix.generator import Generator

# ==========================================
# Device
# ==========================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("Pipeline Inference")
print("=" * 60)
print("Device :", device)

# ==========================================
# Paths
# ==========================================

INPUT_FILE = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "train_sr",
    "input",
    "input_0.npy"
)

SR_MODEL = os.path.join(
    PROJECT_ROOT,
    "models",
    "edsr_x2_best.pth"
)

PIX2PIX_MODEL = os.path.join(
    PROJECT_ROOT,
    "models",
    "pix2pix_generator.pth"
)

RESULT_DIR = os.path.join(
    PROJECT_ROOT,
    "results"
)

os.makedirs(RESULT_DIR, exist_ok=True)

print("Input :", INPUT_FILE)
print("SR Model :", SR_MODEL)
print("Pix2Pix :", PIX2PIX_MODEL)