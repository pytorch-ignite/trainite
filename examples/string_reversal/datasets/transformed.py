from typing import Any

from torch.utils.data import Dataset


class TransformedDataset(Dataset):
    def __init__(self, dataset: Dataset, transform: Any = None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        item = self.dataset[index]
        return self.transform(item) if self.transform else item
