import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder


class HAVSDataset(Dataset):  # Human Activity, Vehicle and Sphere (HAVS)
    """Create a custom dataset"""

    def __init__(self, data, targets, transform=None):
        self.data = data
        self.transform = transform
        self.enc = LabelEncoder()
        targets = self.enc.fit_transform(
            targets.reshape(
                -1,
            )
        )
        self.targets = torch.LongTensor(targets)

    def __getitem__(self, index):  # Memory efficient way of getting items
        if torch.is_tensor(index):
            index = int(index.item())

        x = self.data[index]
        y = self.targets[index]

        if self.transform:
            x = self.data[index]
            x = self.transform(x)  # Transform data

        return x, y

    def __len__(self):
        return len(self.data)
