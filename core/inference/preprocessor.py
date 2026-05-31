# Core Inference Preprocessor
# Đây là module xử lý trước khi đưa dữ liệu vào pipeline chính.
# core/inference/preprocessor.py

# core/inference/preprocessor.py
import numpy as np
import torch
from typing import List

class SequencePreprocessor:
    def __init__(self, seq_len: int = 30, device: str = "cpu"):
        self.seq_len = seq_len
        self.device = device

    def __call__(self, seq_kp: List[np.ndarray]) -> torch.Tensor:
        """Biến đổi danh sách keypoints thành Tensor 68 features đưa vào LSTM"""
        T = len(seq_kp)
        arr = np.zeros((T, 17, 3), dtype=np.float32)
        for i in range(T): arr[i] = seq_kp[i]

        seq_xy = arr[:, :, :2]
        center = seq_xy.mean(axis=(0, 1), keepdims=True) 
        seq_rel = seq_xy - center
        
        scale = np.linalg.norm(seq_rel, axis=2).max()
        if scale > 0: seq_rel /= scale

        velocity = np.zeros_like(seq_rel)
        if T >= 2:
            velocity[1:] = seq_rel[1:] - seq_rel[:-1]
            velocity[0] = velocity[1]

        if T < self.seq_len:
            pad_len = self.seq_len - T
            seq_rel = np.concatenate([seq_rel, np.repeat(seq_rel[-1:], pad_len, axis=0)], axis=0)
            velocity = np.concatenate([velocity, np.zeros((pad_len, 17, 2))], axis=0)
        else:
            seq_rel = seq_rel[-self.seq_len:]
            velocity = velocity[-self.seq_len:]

        seq_xy_flat = seq_rel.reshape(self.seq_len, -1)
        vel_flat = velocity.reshape(self.seq_len, -1)
        seq_flat = np.concatenate([seq_xy_flat, vel_flat], axis=1)

        return torch.from_numpy(seq_flat).float().unsqueeze(0).to(self.device)