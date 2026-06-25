import os
import numpy as np
import torch
from torch.utils.data import Dataset


class SRDataset(Dataset):

    def __init__(self, input_dir, target_dir):

        self.input_dir = input_dir
        self.target_dir = target_dir

        self.input_images = sorted([
            f for f in os.listdir(input_dir)
            if f.endswith(".npy")
        ])

        self.target_images = sorted([
            f for f in os.listdir(target_dir)
            if f.endswith(".npy")
        ])

    def __len__(self):
        return len(self.input_images)

    def __getitem__(self, idx):

        input_name = self.input_images[idx]
        target_name = self.target_images[idx]

        lr = np.load(
            os.path.join(self.input_dir, input_name)
        ).astype(np.float32)

        hr = np.load(
            os.path.join(self.target_dir, target_name)
        ).astype(np.float32)

        # Convert to Torch Tensor
        lr = torch.from_numpy(lr).permute(2, 0, 1)
        hr = torch.from_numpy(hr).permute(2, 0, 1)

        return lr, hr