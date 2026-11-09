import os
import glob
import random
import numpy as np
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


class DailyActionDVS(Dataset):
    """
    Tonic-compatible dataset wrapper for the DailyAction-DVS dataset (128x128, 12 classes).
    Uses a deterministic 90% train+val (`train=True`) / 10% test (`train=False`) split per class,
    yielding an 80-10-10 train-val-test split when combined with a 90/10 train/val random_split.
    """
    sensor_size = (128, 128, 2)
    classes = [
        "bend", "climb", "fall", "get up", "jump", "lie",
        "lift", "pick", "run", "sit", "stand", "walk"
    ]
    num_classes = 12
    EVENT_DTYPE = np.dtype([("x", np.int16), ("y", np.int16), ("t", np.int64), ("p", np.int8)])

    def __init__(self, save_to="./data", train=True, transform=None, test_ratio=0.1, split_seed=42):
        super().__init__()
        self.save_to = save_to
        self.train = train
        self.transform = transform
        self.test_ratio = test_ratio
        self.split_seed = split_seed

        candidate_dir = os.path.join(save_to, "DailyAction-DVS")
        self.root_dir = candidate_dir if os.path.isdir(candidate_dir) else save_to
        if not os.path.isdir(self.root_dir):
            raise FileNotFoundError(
                f"DailyAction-DVS directory not found at '{candidate_dir}' or '{save_to}'."
            )

        self.samples = []
        rng = random.Random(split_seed)

        for label_idx, class_name in enumerate(self.classes):
            class_dir = os.path.join(self.root_dir, class_name)
            if not os.path.isdir(class_dir):
                raise FileNotFoundError(f"Missing class directory: '{class_dir}'")

            files = sorted(glob.glob(os.path.join(class_dir, "*.aedat")))
            rng.shuffle(files)

            n_test = max(1, int(round(len(files) * self.test_ratio)))
            selected_files = files[:-n_test] if self.train else files[-n_test:]
            for fpath in selected_files:
                self.samples.append((fpath, label_idx))

    def __len__(self):
        return len(self.samples)

    @classmethod
    def _read_aedat20(cls, filepath):
        with open(filepath, "rb") as f:
            data = f.read()

        offset = 0
        while offset < len(data) and data[offset:offset + 1] == b"#":
            nl = data.find(b"\n", offset)
            if nl == -1:
                break
            offset = nl + 1

        n_events = (len(data) - offset) // 8
        raw = np.frombuffer(data, dtype=">u4", count=2 * n_events, offset=offset)
        addr = raw[0::2]
        ts = raw[1::2].astype(np.int64)

        # Unwrap 32-bit unsigned microsecond counter overflow (e.g. pick/xd12216.aedat)
        diffs = np.diff(ts)
        wraps = np.where(diffs < -(1 << 30))[0]
        for w in wraps:
            ts[w + 1:] += (1 << 32)

        x = ((addr >> 1) & 0x7F).astype(np.int16)
        y = ((addr >> 8) & 0x7F).astype(np.int16)
        p = (1 - (addr & 1)).astype(np.int8)

        # Ensure strictly non-decreasing timestamps (e.g. minor USB packet jitter in lie/hk6213.aedat)
        if np.any(np.diff(ts) < 0):
            order = np.argsort(ts, kind="stable")
            x, y, ts, p = x[order], y[order], ts[order], p[order]

        if len(ts) > 0:
            ts = ts - ts[0]

        events = np.empty(len(ts), dtype=cls.EVENT_DTYPE)
        events["x"] = x
        events["y"] = y
        events["t"] = ts
        events["p"] = p
        return events

    def __getitem__(self, idx):
        filepath, target = self.samples[idx]
        events = self._read_aedat20(filepath)
        if self.transform is not None:
            events = self.transform(events)
        return events, target
