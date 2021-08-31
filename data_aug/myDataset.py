import torch
from torch.utils.data import Dataset
import numpy as np

class mydataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.df.iloc[idx].to_numpy()

        if self.transform:
            sample = self.transform(sample)

        return sample