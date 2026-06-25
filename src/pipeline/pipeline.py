import os
import sys
import torch

# ==========================================
# Add Project Root to Python Path
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

# ==========================================
# Model Paths
# ==========================================

SR_MODEL = os.path.join(
    PROJECT_ROOT,
    "models",
    "edsr_x2_best.pth"
)

COLOR_MODEL = os.path.join(
    PROJECT_ROOT,
    "models",
    "pix2pix_generator.pth"
)

# ==========================================
# Load Super Resolution Model
# ==========================================

sr_model = EDSR().to(device)

# ==========================================
# Load Colorization Model
# ==========================================

color_model = Generator().to(device)

print("=" * 60)
print("Pipeline Initialized Successfully")
print("=" * 60)

print("Device :", device)
print("SR Model :", SR_MODEL)
print("Color Model :", COLOR_MODEL)