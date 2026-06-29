from typing import Any

from torch.utils.data import Dataset


class TransformedDataset(Dataset):
    def __init__(self, dataset: Dataset, transform: Any):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.transform(self.dataset[index])
