# src/data/dataset_skeleton.py
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
import torch
import random
from typing import List

class SkeletonDataset(Dataset):
    def __init__(self,
                 root_dir: str,
                 classes: List[str],
                 split: str = "train",
                 seq_len: int = 20,
                 augment: bool = False):
        self.root_dir = Path(root_dir)
        self.classes = classes
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        self.seq_len = seq_len
        self.augment = augment

        all_files = []
        for cls in classes:
            class_dir = self.root_dir / cls
            if not class_dir.exists():
                continue
            for f in class_dir.glob("*.npy"):
                all_files.append((f, self.class_to_idx[cls]))

        random.seed(42)
        random.shuffle(all_files)

        n = len(all_files)
        n_train = int(0.8 * n)
        n_val = int(0.1 * n)

        if split == "train":
            self.samples = all_files[:n_train]
        elif split == "val":
            self.samples = all_files[n_train:n_train + n_val]
        else:
            self.samples = all_files[n_train + n_val:]

        print(f"SkeletonDataset[{split}] -> {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        seq = np.load(path)  # (T,17,3)
        T = seq.shape[0]

        # crop / pad về seq_len
        if T > self.seq_len:
            start = random.randint(0, T - self.seq_len)
            seq = seq[start:start+self.seq_len]
        elif T < self.seq_len:
            pad = np.repeat(seq[-1][None, :, :], self.seq_len - T, axis=0)
            seq = np.concatenate([seq, pad], axis=0)

        seq_xy = seq[:, :, :2]
        center = seq_xy.mean(axis=1, keepdims=True)
        seq_rel = seq_xy - center

        scale = np.linalg.norm(seq_rel, axis=2).max()
        if scale > 0:
            seq_rel /= scale

        seq_flat = seq_rel.reshape(self.seq_len, -1)

        x = torch.from_numpy(seq_flat).float()
        y = torch.tensor(label, dtype=torch.long)
        return x, y
