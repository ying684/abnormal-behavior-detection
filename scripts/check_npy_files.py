#!/usr/bin/env python3
from pathlib import Path
import numpy as np

root = Path("data/processed/sequences")

print("=== CHECKING .NPY FILES ===\n")

for cls in ["normal", "fighting", "falling", "loitering"]:
    cls_dir = root / cls
    if not cls_dir.exists():
        print(f"❌ {cls}: dir not found")
        continue
    
    npy_files = list(cls_dir.glob("*.npy"))
    print(f"\n📁 {cls}: {len(npy_files)} files")
    
    if npy_files:
        # Check first 3 files
        for npy_file in npy_files[:3]:
            data = np.load(npy_file)
            min_val, max_val, mean_val = data.min(), data.max(), data.mean()
            print(f"  {npy_file.name}")
            print(f"    shape: {data.shape}, dtype: {data.dtype}")
            print(f"    min: {min_val:.4f}, max: {max_val:.4f}, mean: {mean_val:.4f}")
            if np.all(data == 0):
                print(f"    ⚠️ ALL ZEROS!")
