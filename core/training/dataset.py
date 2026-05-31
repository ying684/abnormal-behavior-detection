# core/training/dataset.py

from collections import defaultdict
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
import torch
import random
from typing import List

class SkeletonDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        classes: List[str],
        split: str = "train",
        seq_len: int = 30,
        augment: bool = False
    ):
        self.root_dir = Path(root_dir)
        self.classes = classes
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        self.seq_len = seq_len
        self.augment = augment

        video_groups = defaultdict(list)
        for cls in classes:
            class_dir = self.root_dir / cls
            if not class_dir.exists(): continue
            for f in class_dir.glob("*.npy"):
                video_name = f.stem.rsplit("_id", 1)[0]
                video_groups[(cls, video_name)].append((f, self.class_to_idx[cls]))

        self.samples = []
        
        # FIX LỖI 2: Chia 80/20 ĐỀU CHO TỪNG CLASS (Stratified Split)
        for cls in classes:
            # Lọc ra các video thuộc class hiện tại
            cls_keys = [k for k in video_groups.keys() if k[0] == cls]
            random.seed(42)
            random.shuffle(cls_keys)

            n = len(cls_keys)
            n_train = int(0.8 * n)

            if split == "train":
                selected = cls_keys[:n_train]
            elif split == "val":
                selected = cls_keys[n_train:]          

            for key in selected:
                self.samples.extend(video_groups[key])

        print(f"SkeletonDataset[{split}] -> {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        seq = np.load(path)  
        T = seq.shape[0]

        if T > self.seq_len:
            start = random.randint(0, T - self.seq_len)
            seq = seq[start : start + self.seq_len]
        elif T < self.seq_len:
            pad = np.repeat(seq[-1][None, :, :], self.seq_len - T, axis=0)
            seq = np.concatenate([seq, pad], axis=0)

        seq_xy = seq[:, :, :2]

        # FIX LỖI 1: TÍNH TÂM CỦA TOÀN BỘ SEQUENCE (Giữ lại đặc trưng di chuyển)
        global_center = seq_xy.mean(axis=(0, 1), keepdims=True) # shape (1, 1, 2)
        seq_rel = seq_xy - global_center

        scale = np.linalg.norm(seq_rel, axis=2).max()
        if scale > 0:
            seq_rel /= scale

        if self.augment:
            noise = np.random.normal(0, 0.01, seq_rel.shape)
            seq_rel += noise
            
            scale_aug = np.random.uniform(0.95, 1.05)
            seq_rel *= scale_aug
            
            shift_x = np.random.uniform(-0.02, 0.02)
            shift_y = np.random.uniform(-0.02, 0.02)
            seq_rel[:, :, 0] += shift_x
            seq_rel[:, :, 1] += shift_y
            
            # LẬT NGANG (HORIZONTAL FLIP) - Nhúp tăng cường dữ liệu cho các tư thế đảo chiều
            if random.random() > 0.5:
                seq_rel[:, :, 0] = -seq_rel[:, :, 0]

        velocity = np.zeros_like(seq_rel)
        if self.seq_len > 1:
            velocity[1:] = seq_rel[1:] - seq_rel[:-1]
            velocity[0] = velocity[1]

        seq_xy_flat = seq_rel.reshape(self.seq_len, -1)
        vel_flat = velocity.reshape(self.seq_len, -1)
        seq_flat = np.concatenate([seq_xy_flat, vel_flat], axis=1)

        x = torch.from_numpy(seq_flat).float()
        y = torch.tensor(label, dtype=torch.long)
        return x, y