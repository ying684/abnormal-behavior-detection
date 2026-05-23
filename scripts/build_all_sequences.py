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

    for cls in classes:
        class_dir = root_raw / cls
        if not class_dir.exists():
            print(f"[WARN] class dir not found: {class_dir}")
            continue

        out_dir = Path("data/processed/sequences") / cls
        out_dir.mkdir(parents=True, exist_ok=True)

        for video_file in class_dir.glob("*.mp4"):
            print(f"[{cls}] Processing {video_file.name}")
            builder.build_sequences_for_video(
                video_path=str(video_file),
                save_dir=str(out_dir),
                min_len=20
            )

if __name__ == "__main__":
    main()
