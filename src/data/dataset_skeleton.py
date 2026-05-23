# src/data/dataset_skeleton.py
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
import torch
import random
from typing import List

class SkeletonDataset(Dataset):
    def __init__(self, root_dir: str, classes: List[str], split: str = "train", seq_len: int = 30, augment: bool = False):
        self.root_dir = Path(root_dir)
        self.classes = classes
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        self.seq_len = seq_len
        self.augment = augment
        self.split = split

        all_files = []
        for cls in classes:
            class_dir = self.root_dir / cls
            if not class_dir.exists(): continue
            for f in class_dir.glob("*.npy"):
                all_files.append((f, self.class_to_idx[cls]))

        random.seed(42)
        random.shuffle(all_files)

        n = len(all_files)
        n_train = int(0.8 * n)
        n_val = int(0.1 * n)

        if split == "train": self.samples = all_files[:n_train]
        elif split == "val": self.samples = all_files[n_train:n_train+n_val]
        else: self.samples = all_files[n_train+n_val:]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        seq = np.load(path)  # (T,17,3)
        T = seq.shape[0]

        # Padding / Cropping
        if T > self.seq_len:
            start = random.randint(0, T - self.seq_len)
            seq = seq[start:start+self.seq_len]
        elif T < self.seq_len:
            pad = np.repeat(seq[-1][None, :, :], self.seq_len - T, axis=0)
            seq = np.concatenate([seq, pad], axis=0)

        seq_xy = seq[:, :, :2]  # Lấy x, y

        # Augmentation: Thêm nhiễu ngẫu nhiên vào toạ độ (chỉ áp dụng cho tập train)
        if self.augment and self.split == "train":
            noise = np.random.normal(0, 0.01, seq_xy.shape)
            seq_xy = seq_xy + noise

        # Chuẩn hoá toạ độ tương đối
        center = seq_xy.mean(axis=1, keepdims=True)
        seq_rel = seq_xy - center
        scale = np.linalg.norm(seq_rel, axis=2).max()
        if scale > 0:
            seq_rel /= scale

        seq_flat = seq_rel.reshape(self.seq_len, -1)  # (seq_len, 34)

        # Tính toán Vận tốc (Velocity) giữa các frame
        velocity = np.zeros_like(seq_flat)
        velocity[1:] = seq_flat[1:] - seq_flat[:-1]

        # Ghép Tọa độ và Vận tốc -> vector 68 chiều
        features = np.concatenate([seq_flat, velocity], axis=1) # (seq_len, 68)

        x = torch.from_numpy(features).float()
        y = torch.tensor(label, dtype=torch.long)
        return x, y