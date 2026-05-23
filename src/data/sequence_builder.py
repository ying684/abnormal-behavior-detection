# src/data/sequence_builder.py
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO
import torch

class FastSequenceBuilder:
    """
    Tối ưu build skeleton sequences:
    - Batch inference với YOLO
    - Skip frame (lấy 1 frame mỗi N frame)
    - Multiprocessing
    """

    def __init__(self, pose_weight: str, device: str = "cuda"):
        self.model = YOLO(pose_weight)
        self.model.to(device)
        self.device = device

    def build_sequence_for_video(self,
                                  video_path: str,
                                  save_path: str,
                                  frame_skip: int = 2,
                                  max_frames: int = 150,
                                  min_len: int = 15):
        """
        frame_skip: lấy 1 frame mỗi N frame (skip=2 -> 1,3,5,7...)
        max_frames: tối đa bao nhiêu frame sau khi skip
        min_len: độ dài tối thiểu sequence
        
        Với skip=2, max_frames=150:
        - Tối đa 300 frame gốc
        - Khoảng 10 giây @30fps
        - Đủ thấy hành vi
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return False

        frames = []
        frame_idx = 0

        # Lấy frames (skip)
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_skip == 0:
                frames.append(frame)
                if len(frames) >= max_frames:
                    break
            frame_idx += 1

        cap.release()

        if len(frames) < min_len:
            return False

        # Batch inference YOLO-Pose
        keypoints_list = []
        batch_size = 8  # xử lý 8 frame cùng lúc

        for batch_start in range(0, len(frames), batch_size):
            batch_frames = frames[batch_start:batch_start + batch_size]

            results = self.model.predict(
                batch_frames,
                device=self.device,
                verbose=False,
                conf=0.5,
                iou=0.45
            )

            for result in results:
                if result.keypoints is None or result.boxes is None:
                    # Nếu frame này không detect ai, dùng frame trước
                    if keypoints_list:
                        keypoints_list.append(keypoints_list[-1])
                    continue

                boxes = result.boxes.xyxy.cpu().numpy()
                kps_xy = result.keypoints.xy.cpu().numpy()
                kps_conf = result.keypoints.conf.cpu().numpy()

                if len(boxes) == 0:
                    if keypoints_list:
                        keypoints_list.append(keypoints_list[-1])
                    continue

                # Chọn person lớn nhất
                areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                main_idx = np.argmax(areas)

                kp_xy = kps_xy[main_idx]
                kp_c = kps_conf[main_idx][:, None]
                keypoints = np.concatenate([kp_xy, kp_c], axis=1)
                keypoints_list.append(keypoints)

        if len(keypoints_list) < min_len:
            return False

        seq = np.stack(keypoints_list, axis=0)  # (T,17,3)
        np.save(save_path, seq)
        return True

    def build_for_class_folder(self,
                               class_dir: str,
                               out_dir: str,
                               frame_skip: int = 2,
                               max_frames: int = 150,
                               min_len: int = 15,
                               n_workers: int = 4):
        """
        n_workers: số process chạy song song
        """
        from concurrent.futures import ProcessPoolExecutor, as_completed

        class_dir = Path(class_dir)
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        video_files = list(class_dir.glob("*.mp4"))
        if not video_files:
            print(f"[WARN] No videos in {class_dir}")
            return

        print(f"[{class_dir.name}] {len(video_files)} videos, {n_workers} workers")

        def process_video(vf):
            out_path = out_dir / (vf.stem + ".npy")
            success = self.build_sequence_for_video(
                str(vf),
                str(out_path),
                frame_skip=frame_skip,
                max_frames=max_frames,
                min_len=min_len
            )
            return vf.name, success

        # Multiprocessing
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(process_video, vf): vf
                for vf in video_files
            }
            for future in tqdm(as_completed(futures), total=len(futures), desc=class_dir.name):
                try:
                    fname, success = future.result()
                    if success:
                        print(f"✓ {fname}")
                    else:
                        print(f"✗ {fname} (skipped - too short)")
                except Exception as e:
                    print(f"Error: {e}")
