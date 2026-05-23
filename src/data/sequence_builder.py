# src/data/sequence_builder.py
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict

from src.detection.tracker import PersonTracker
from src.detection.pose_estimator import PoseEstimator

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

class SequenceBuilder:
    def __init__(self, yolo_weight: str, pose_weight: str, device: str = "cuda"):
        self.tracker = PersonTracker(yolo_weight, device)
        self.pose_model = PoseEstimator(pose_weight, device)

    def build_sequences_for_video(self, video_path: str, save_dir: str, min_len: int = 20):
        """
        - Track persons
        - Pose estimation
        - Gán keypoints cho mỗi track_id
        - Lưu mỗi (video, track_id) -> .npy (T,17,3)
        """
        video_path = Path(video_path)
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"[WARN] Cannot open video {video_path}")
            return

        all_tracks = defaultdict(list)  # track_id -> list[keypoints or None]
        track_gen = self.tracker.track_stream(str(video_path))

        for dets in track_gen:
            ret, frame = cap.read()
            if not ret:
                break

            persons_pose = self.pose_model.estimate(frame)

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

                if best_kp is not None and best_i > 0.5:
                    all_tracks[tid].append(best_kp)
                else:
                    all_tracks[tid].append(None)

        cap.release()

        for tid, kp_list in all_tracks.items():
            T = len(kp_list)
            if T < min_len:
                continue

            # Count tracking losses (None values)
            none_count = sum(1 for kp in kp_list if kp is None)
            loss_rate = none_count / T
            
            # Skip sequences with >30% tracking loss (bad quality)
            if loss_rate > 0.3:
                print(f"  ⚠️ Skipped sequence (loss_rate={loss_rate:.1%} > 30%), length={T}")
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
                    else:
                        kp_array[t] = 0.0

            out_name = f"{video_path.stem}_id{tid}.npy"
            np.save(save_dir / out_name, kp_array)
            print(f"Saved sequence {save_dir/out_name}, length={T}")
