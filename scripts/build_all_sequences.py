# scripts/build_all_sequences.py
import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.sequence_builder import FastSequenceBuilder

def main():
    root_raw = Path("data/raw")
    classes = ["normal", "fighting", "falling", "loitering"]

    builder = FastSequenceBuilder(
        pose_weight="weights/yolov8s-pose.pt",
        device="cuda"
    )

    for cls in classes:
        class_dir = root_raw / cls
        if not class_dir.exists():
            print(f"[WARN] class dir not found: {class_dir}")
            continue

        out_dir = Path("data/processed/sequences") / cls

        print(f"\n=== Building {cls} ===")
        builder.build_for_class_folder(
            class_dir=str(class_dir),
            out_dir=str(out_dir),
            frame_skip=2,           # lấy 1 frame mỗi 2 frame (giảm 50% từ số frame)
            max_frames=150,         # tối đa 150 frame sau skip = khoảng 10 giây
            min_len=15,             # tối thiểu 15 frame (~1 giây)
            n_workers=4             # 4 video song song
        )

if __name__ == "__main__":
    main()
