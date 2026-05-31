# scripts/data_processing/build_sequences.py

from pathlib import Path
import sys
import cv2
import numpy as np
from collections import defaultdict

sys.path.append(str(Path(__file__).parent.parent.parent))
from core.detection.pose_tracker import PoseTracker
from config.settings import settings

def build_sequences_for_video(video_path: Path, save_dir: Path, tracker: PoseTracker, min_len: int = 30) -> bool:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return False
    
    all_tracks = defaultdict(list)
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        persons = tracker.track_frame(frame)
        for p in persons:
            tid = p["track_id"]
            # Lưu keypoints nếu pose đủ tự tin (ví dụ: conf trung bình của các điểm > 0.5)
            avg_kp_conf = np.mean(p["keypoints"][:, 2])
            if avg_kp_conf > 0.3:
                all_tracks[tid].append(p["keypoints"])
            else:
                all_tracks[tid].append(None)
                
    cap.release()
    valid_sequence_count = 0
    
    for tid, kp_list in all_tracks.items():
        T = len(kp_list)
        if T < min_len: continue
            
        valid_frames = sum(1 for kp in kp_list if kp is not None)
        if valid_frames / T < 0.6: continue # Lọc rác (tỉ lệ pose hợp lệ < 60%)
        
        kp_array = np.zeros((T, 17, 3), dtype=np.float32)
        last_valid = None
        for t in range(T):
            if kp_list[t] is not None:
                kp_array[t] = kp_list[t]
                last_valid = kp_list[t]
            elif last_valid is not None:
                kp_array[t] = last_valid
                
        out_name = f"{video_path.stem}_id{tid}.npy"
        np.save(save_dir / out_name, kp_array)
        valid_sequence_count += 1

    return valid_sequence_count > 0

def main():
    root_raw = settings.data_dir / "raw"
    # CHỈ BUILD SEQUENCE CHO 3 CLASS NÀY:
    target_classes = ["normal", "fighting", "falling"]
    
    tracker = PoseTracker(
        str(settings.weights_dir / "detection/yolov8s-pose.pt"),
        settings.device
    )

    print("=" * 50)
    print("BUILDING SKELETON SEQUENCES (EXCLUDING LOITERING)")
    print("=" * 50)

    for cls in target_classes:
        class_dir = root_raw / cls
        if not class_dir.exists(): continue

        out_dir = settings.data_dir / "processed" / "sequences" / cls
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n[{cls.upper()}]")

        for video_file in class_dir.glob("*.mp4"):
            if len(list(out_dir.glob(f"{video_file.stem}_id*.npy"))) > 0:
                continue
            
            build_sequences_for_video(video_file, out_dir, tracker, min_len=30)
            print(f"  Processed: {video_file.name}")

if __name__ == "__main__":
    main()