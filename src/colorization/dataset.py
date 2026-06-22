import os
import cv2
import torch
from torch.utils.data import Dataset

class LandsatDataset(Dataset):

    def __init__(self,
                 input_dir,
                 rgb_dir):

        self.input_dir = input_dir
        self.rgb_dir = rgb_dir

        self.input_files = sorted(
            os.listdir(input_dir)
        )

        self.rgb_files = sorted(
            os.listdir(rgb_dir)
        )

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, idx):

        input_path = os.path.join(
            self.input_dir,
            self.input_files[idx]
        )

        rgb_path = os.path.join(
            self.rgb_dir,
            self.rgb_files[idx]
        )

        input_img = cv2.imread(input_path)
        rgb_img = cv2.imread(rgb_path)

        input_img = input_img / 255.0
        rgb_img = rgb_img / 255.0

        input_img = torch.tensor(
            input_img,
            dtype=torch.float32
        ).permute(2,0,1)

        rgb_img = torch.tensor(
            rgb_img,
            dtype=torch.float32
        ).permute(2,0,1)

        return input_img, rgb_img