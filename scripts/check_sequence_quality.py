#!/usr/bin/env python3
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset_skeleton import SkeletonDataset

classes = ["normal", "fighting", "falling", "loitering"]

print("=== SEQUENCE QUALITY CHECK ===\n")

train_ds = SkeletonDataset(
    root_dir="data/processed/sequences",
    classes=classes,
    split="train",
    seq_len=20
)

# Check variance per class
class_stats = {cls: {"variances": [], "max_vals": []} for cls in classes}

for i in range(min(len(train_ds), 100)):  # Sample first 100
    x, y = train_ds[i]
    cls_name = classes[y]
    
    # Compute per-sequence variance (should be high for good features)
    var = np.var(x.numpy())
    max_val = np.max(np.abs(x.numpy()))
    
    class_stats[cls_name]["variances"].append(var)
    class_stats[cls_name]["max_vals"].append(max_val)

print("📊 VARIANCE BY CLASS (higher = more movement/variation):")
for cls, stats in class_stats.items():
    if stats["variances"]:
        avg_var = np.mean(stats["variances"])
        avg_max = np.mean(stats["max_vals"])
        print(f"  {cls:12} : var={avg_var:.4f}, max={avg_max:.4f}")

print("\n💡 If variances are similar across classes → sequences too similar → hard to distinguish")
print("💡 If max values are low (<0.3) → not enough movement/variation in poses")
