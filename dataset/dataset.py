import os
import torch
from torch.utils.data import Dataset

class DVSGestureData(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        event, target = self.dataset[idx]
        return self._transform(event, target)

    def _transform(self, event, target):
        event = torch.tensor(event.transpose(1, 0, 2, 3), dtype=torch.float32)
        target = torch.tensor(target, dtype=torch.int64)
        return event, target

class CachedDVSGestureData(Dataset):
    def __init__(self, cache_path):
        self.cache_path = cache_path

    def __len__(self):
        return len(os.listdir(self.cache_path))

    def __getitem__(self, idx):
        cached_path = os.path.join(self.cache_path, f"sample_{idx}.pt")
        data = torch.load(cached_path, weights_only=False)
        event, target = data
        return self._transform(event, target)

    def _transform(self, event, target):
        event = torch.tensor(event.transpose(1, 0, 2, 3), dtype=torch.float32)
        target = torch.tensor(target, dtype=torch.int64)
        return event, target