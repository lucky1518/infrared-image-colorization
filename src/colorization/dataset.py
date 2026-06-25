import os
import numpy as np
import torch
from torch.utils.data import Dataset


class LandsatDataset(Dataset):

    def __init__(self, input_dir, rgb_dir):

        self.input_dir = input_dir
        self.rgb_dir = rgb_dir

        # Load all .npy input files
        self.input_images = sorted(
            [
                f for f in os.listdir(input_dir)
                if f.endswith(".npy")
            ]
        )

    def __len__(self):
        return len(self.input_images)

    def __getitem__(self, idx):

        # Input file
        input_name = self.input_images[idx]

        # Matching RGB file
        rgb_name = input_name.replace("input", "rgb")

        input_path = os.path.join(self.input_dir, input_name)
        rgb_path = os.path.join(self.rgb_dir, rgb_name)

        # Load NPY files
        input_img = np.load(input_path).astype(np.float32)
        rgb_img = np.load(rgb_path).astype(np.float32)

        # Normalize each patch to [0,1]
        input_img = input_img / input_img.max()
        rgb_img = rgb_img / rgb_img.max()

        # Convert HWC -> CHW
        input_img = torch.from_numpy(
            input_img.transpose(2, 0, 1)
        )

        rgb_img = torch.from_numpy(
            rgb_img.transpose(2, 0, 1)
        )

        return input_img, rgb_img