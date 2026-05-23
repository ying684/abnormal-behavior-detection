# scripts/build_all_sequences.py
import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.sequence_builder import SequenceBuilder

def main():
    root_raw = Path("data/raw")
    classes = ["normal", "fighting", "falling", "loitering"]

    builder = SequenceBuilder(
        yolo_weight="weights/yolov8s.pt",
        pose_weight="weights/yolov8s-pose.pt",
        device="cuda"
    )

    max_videos_per_class = 15  # Balance: 60 videos, ~350 samples, ~40 phút train
    
    for cls in classes:
        class_dir = root_raw / cls
        if not class_dir.exists():
            print(f"[WARN] class dir not found: {class_dir}")
            continue

        out_dir = Path("data/processed/sequences") / cls
        out_dir.mkdir(parents=True, exist_ok=True)

        video_files = list(class_dir.glob("*.mp4"))[:max_videos_per_class]
        print(f"[{cls}] Processing {len(video_files)}/{len(class_dir.glob('*.mp4'))} videos")
        
        for video_file in video_files:
            print(f"[{cls}] Processing {video_file.name}")
            builder.build_sequences_for_video(
                video_path=str(video_file),
                save_dir=str(out_dir),
                min_len=20
                # Bỏ max_len để giữ toàn bộ info, nhưng chia nhỏ sequences
            )

if __name__ == "__main__":
    main()
