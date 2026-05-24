# scripts/data_processing/build_sequences.py
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from core.detection.tracker import PersonTracker
from core.detection.pose_estimator import PoseEstimator
from config.settings import settings
import cv2
import numpy as np
from collections import defaultdict

def _iou(b1, b2):
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])
    inter = max(0, x2-x1) * max(0, y2-y1)
    a1 = max(0, b1[2]-b1[0]) * max(0, b1[3]-b1[1])
    a2 = max(0, b2[2]-b2[0]) * max(0, b2[3]-b2[1])
    union = a1 + a2 - inter
    if union <= 0:
        return 0.0
    return inter / union

def build_sequences_for_video(video_path: str, save_dir: str, min_len: int = 20):
    video_path = Path(video_path)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    tracker = PersonTracker(
        str(settings.weights_dir / "detection/yolov8s.pt"),
        settings.device
    )
    pose_model = PoseEstimator(
        str(settings.weights_dir / "detection/yolov8s-pose.pt"),
        settings.device
    )
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Cannot open video {video_path}")
        return
    
    all_tracks = defaultdict(list)
    track_gen = tracker.track_stream(str(video_path))
    
    for dets in track_gen:
        ret, frame = cap.read()
        if not ret:
            break
        
        persons_pose = pose_model.estimate(frame)
        
        for d in dets:
            tid = d["track_id"]
            bbox = d["bbox"]
            best_kp = None
            best_i = 0.0
            for p in persons_pose:
                i = _iou(bbox, p["bbox"])
                if i > best_i:
                    best_i = i
                    best_kp = p["keypoints"]
            
            if best_kp is not None and best_i > 0.3:
                all_tracks[tid].append(best_kp)
            else:
                all_tracks[tid].append(None)
    
    cap.release()
    
    for tid, kp_list in all_tracks.items():
        T = len(kp_list)
        if T < min_len:
            continue
        
        kp_array = np.zeros((T, 17, 3), dtype=np.float32)
        last_valid = None
        for t in range(T):
            if kp_list[t] is not None:
                kp_array[t] = kp_list[t]
                last_valid = kp_list[t]
            else:
                if last_valid is not None:
                    kp_array[t] = last_valid
        
        out_name = f"{video_path.stem}_id{tid}.npy"
        np.save(save_dir / out_name, kp_array)
        print(f"Saved {save_dir/out_name}, length={T}")

def main():
    root_raw = settings.data_dir / "raw"
    classes = settings.classes
    
    for cls in classes:
        class_dir = root_raw / cls
        if not class_dir.exists():
            print(f"Class dir not found: {class_dir}")
            continue
        
        out_dir = settings.data_dir / "processed/sequences" / cls
        out_dir.mkdir(parents=True, exist_ok=True)
        
        for video_file in class_dir.glob("*.mp4"):
            print(f"[{cls}] Processing {video_file.name}")
            build_sequences_for_video(
                str(video_file),
                str(out_dir),
                min_len=20
            )

if __name__ == "__main__":
    main()
