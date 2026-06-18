from torch.utils.data import Dataset
from trainite.datasets.transformed import TransformedDataset


class StubDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class StubTransform:
    def __call__(self, sample):
        return f"transformed_{sample}"


def test_transformed_dataset():
    raw_data = ["apple", "banana", "cherry"]
    dataset = StubDataset(raw_data)
    transform = StubTransform()
    transformed_dataset = TransformedDataset(dataset, transform)

    assert len(transformed_dataset) == 3
    assert transformed_dataset[0] == "transformed_apple"
    assert transformed_dataset[1] == "transformed_banana"
    assert transformed_dataset[2] == "transformed_cherry"
