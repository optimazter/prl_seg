import torch
from torch.utils.data import Dataset
import os

class LazyDataset(Dataset):

    """
    A dataset that lazily loads files from a specified directory.
    This dataset assumes that all files in the directory are PyTorch tensors with a .pt extension.
    Args:
        dir (str): The directory containing the .pt files.
    """

    def __init__(self, dir: str):
        self.files = [os.path.join(dir, f) for f in os.listdir(dir)]
        assert len(self.files) > 0, f"No files found in {dir}!"
        assert all([f.endswith(".pt") for f in self.files]), f"Files in {dir} must have .pt extension!"

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        return torch.load(self.files[idx], weights_only=False)