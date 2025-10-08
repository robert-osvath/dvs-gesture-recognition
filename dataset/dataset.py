import torch
from torch.utils.data import Dataset

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