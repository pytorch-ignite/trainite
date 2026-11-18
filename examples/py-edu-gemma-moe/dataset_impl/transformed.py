from typing import Any

from torch.utils.data import Dataset


class TransformedDataset(Dataset):
    """Wraps a base Dataset and applies an optional callable transform to each item.

    This follows the standard PyTorch ``Dataset`` pattern where a separate
    transform object handles feature extraction / tokenisation, keeping the raw
    data storage (the base dataset) decoupled from preprocessing logic.

    If no transform is provided the raw item is returned unchanged.

    Docs: https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset
    """

    def __init__(self, dataset: Dataset, transform: Any = None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        item = self.dataset[index]
        return self.transform(item) if self.transform else item
