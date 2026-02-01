import os
import tonic
from dataset.dataset import CachedDVSGestureData
from utils.pad_tensors import PadTensorsUpdated
from torch.utils.data import DataLoader, random_split

cached_train_dataset = CachedDVSGestureData(cache_path=os.path.join('./cache', 'time_window', 'train'))
cached_test_dataset = CachedDVSGestureData(cache_path=os.path.join('./cache', 'time_window', 'test'))

train_data, val_data, _ = random_split(
    cached_train_dataset,
    [0.1, 0.05, round((1 - 0.1 - 0.05)*100)/100],
)

test_data = cached_test_dataset

collate_fn=PadTensorsUpdated(batch_first=True, target_frames=800, mode='max')

train_loader = DataLoader(train_data, batch_size=8, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_data, batch_size=8, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_data, batch_size=8, shuffle=False, collate_fn=collate_fn)

for batch_idx, (data, target) in enumerate(train_loader):
    print(f"Batch {batch_idx}: data shape = {data.shape}, target shape = {target.shape}")