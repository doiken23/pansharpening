from pathlib import Path

import numpy as np
import joblib

import torch.utils.data as data_utils

class PansharpenDataset(data_utils.Dataset):
    def __init__(self, data_dir, transforms=None):
        self.data_dir = Path(data_dir)
        self.data_paths = list(self.data_dir.iterdir())
        self.transforms = transforms

    def __getitem__(self, idx):
        data = joblib.load(self.data_paths[idx])
        rgb = data['rgb']
        panchro = data['l8']

        if self.transforms is not None:
            rgb, panchro = self.transforms(rgb, panchro)

        return rgb, panchro
