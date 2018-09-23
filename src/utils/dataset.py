from pathlib import Path

import joblib
import numpy as np

import torch.utils.data as data_utils

class PansharpenDataset(data_utils.Dataset):
    def __init__(self, data, transforms=None):
        self.data = Path(data)
        self.data_paths = list(self.data.iterdir())
        self.transforms = transforms

    def __getitem__(self, idx):
        data_path = str(self.data_paths[idx])
        data = joblib.load(data_path)
        data = [data['rgb'].astype(np.float32), data['b8'].astype(np.float32)]

        rgb, b8 = data
        if len(b8.shape) == 2:
            b8 = b8[np.newaxis, ...]
        data = (rgb, b8)

        if self.transforms is not None:
            data = self.transforms(data)
        return data

    def __len__(self):
        return len(self.data_paths)
