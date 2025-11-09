import os
import torch
from torch.utils.data import Dataset
import os

class DVSGestureData(Dataset):
    def __init__(self, dataset, transform=None):
      self.dataset = dataset
      self.transform = transform

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
    def __init__(self, dataset, cache_path, transform=None):
        self.dataset = dataset
        self.cache_path = cache_path
        self.transform = transform

        if not os.path.exists(self.cache_path):
            os.makedirs(self.cache_path)

        num_cached_files = len(os.listdir(self.cache_path))
        if num_cached_files != len(self.dataset):
            print(f"Cache incomplete or missing. Caching {len(self.dataset)} samples to {self.cache_path}...")
            self._cache_dataset()
        else:
            print(f"Found complete cache at {self.cache_path}.")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        cached_path = os.path.join(self.cache_path, f"sample_{idx}.pt")
        return torch.load(cached_path, weights_only=False)

    def _cache_dataset(self):
        for idx in range(len(self.dataset)):
            event, target = self._transform(*self.dataset[idx])
            
            cached_path = os.path.join(self.cache_path, f"sample_{idx}.pt")
            torch.save((event, target), cached_path)

    def _transform(self, event, target):
        if self.transform:
            event, target = self.transform(event, target)
        event = torch.tensor(event.transpose(1, 0, 2, 3), dtype=torch.float32)
        target = torch.tensor(target, dtype=torch.int64)
        return event, target