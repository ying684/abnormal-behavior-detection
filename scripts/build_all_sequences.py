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

    max_videos_per_class = 25  # Tăng lên để có đủ data, especially loitering
    
    for cls in classes:
        class_dir = root_raw / cls
        if not class_dir.exists():
            print(f"[WARN] class dir not found: {class_dir}")
            continue

        out_dir = Path("data/processed/sequences") / cls
        out_dir.mkdir(parents=True, exist_ok=True)

        all_videos = list(class_dir.glob("*.mp4"))
        video_files = all_videos[:max_videos_per_class]
        
        # Skip already processed videos
        processed_count = 0
        skipped_count = 0
        for video_file in video_files:
            # Check if this video has any sequences already saved
            existing_seqs = list(out_dir.glob(f"{video_file.stem}_*.npy"))
            if existing_seqs:
                skipped_count += 1
                continue
            processed_count += 1
        
        print(f"[{cls}] Processing {processed_count} new / {len(video_files)} total videos (skipped {skipped_count} already processed)")
        
        for video_file in video_files:
            # Skip if already processed
            existing_seqs = list(out_dir.glob(f"{video_file.stem}_*.npy"))
            if existing_seqs:
                continue
                
            print(f"[{cls}] Processing {video_file.name}")
            try:
                builder.build_sequences_for_video(
                    video_path=str(video_file),
                    save_dir=str(out_dir),
                    min_len=20
                    # Bỏ max_len để giữ toàn bộ info, nhưng chia nhỏ sequences
                )
            except KeyboardInterrupt:
                print(f"\n[INTERRUPTED] Saved {sum(len(list(Path('data/processed/sequences') / cls).glob('*.npy')) for cls in classes)} sequences total")
                raise
            except Exception as e:
                print(f"[ERROR] Failed to process {video_file.name}: {e}")

if __name__ == "__main__":
    main()
