#!/usr/bin/env python3
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset_skeleton import SkeletonDataset

classes = ["normal", "fighting", "falling", "loitering"]

print("=== DATASET ANALYSIS ===\n")

# Check train split
train_ds = SkeletonDataset(
    root_dir="data/processed/sequences",
    classes=classes,
    split="train",
    seq_len=20
)

# Check distribution
class_counts = {cls: 0 for cls in classes}
for i in range(len(train_ds)):
    x, y = train_ds[i]
    class_counts[classes[y]] += 1
    
    # Check for NaN/Invalid
    if np.isnan(x).any():
        print(f"⚠️ Sample {i} contains NaN values!")
        break
    if x.shape != (20, 34):
        print(f"⚠️ Sample {i} has wrong shape: {x.shape} (expected (20, 34))")

print("\n📊 CLASS DISTRIBUTION (Training):")
for cls, count in class_counts.items():
    pct = (count / len(train_ds)) * 100
    print(f"  {cls:12} : {count:3d} samples ({pct:5.1f}%)")

print(f"\nTotal train samples: {len(train_ds)}")

# Check a sample
x_sample, y_sample = train_ds[0]
print(f"\n📋 Sample shape: {x_sample.shape}")
print(f"   Sample class: {classes[y_sample]}")
print(f"   Sample min: {x_sample.min():.4f}, max: {x_sample.max():.4f}")
print(f"   Sample mean: {x_sample.mean():.4f}, std: {x_sample.std():.4f}")
