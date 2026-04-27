"""Base class for pyaptamer datasets."""

__author__ = ["siddharth7113", "nennomp"]
__all__ = ["BaseDataset"]

from skbase.base import BaseObject
from torch.utils.data import Dataset


class BaseDataset(BaseObject, Dataset):
    """Base class for PyTorch datasets in pyaptamer."""

    def __init__(self):
        super().__init__()

    def __len__(self):
        """Return length of dataset."""
        raise NotImplementedError("abstract method")

    def __getitem__(self, index):
        """Get item at index."""
        raise NotImplementedError("abstract method")
