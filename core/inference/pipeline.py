# core/inference/pipeline.py
import cv2
import torch
import time
import csv
import numpy as np  # <-- ĐÃ FIX LỖI THIẾU NUMPY
from datetime import datetime
from collections import defaultdict, deque
from typing import List

from core.detection.pose_tracker import PoseTracker
from core.models.lstm_skeleton import SkeletonLSTM
from core.inference.preprocessor import SequencePreprocessor
from core.inference.postprocessor import BehaviorPostprocessor
from config.settings import settings

class RealtimeBehaviorRecognizer:
    def __init__(
        self,
        pose_weight: str,
        lstm_weight: str,
        classes: List[str] = ["normal", "fighting", "falling"],
        device: str = "cuda",
        seq_len: int = 30,
        loitering_threshold_sec: float = 10.0
    ):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.classes = classes
        
        self.tracker = PoseTracker(pose_weight, self.device)
        self.preprocessor = SequencePreprocessor(seq_len=seq_len, device=self.device)
        self.postprocessor = BehaviorPostprocessor(loitering_threshold_sec=loitering_threshold_sec)
        
        # <-- ĐÃ FIX: Khớp 100% với cấu trúc model lúc bạn train (128, 2, 3 class)
        self.model = SkeletonLSTM(
            input_size=68, 
            hidden_size=128, 
            num_layers=2,
            num_classes=3
        ).to(self.device)
        
        try:
            self.model.load_state_dict(torch.load(lstm_weight, map_location=self.device, weights_only=False))
            print(f"✅ Loaded LSTM weights from {lstm_weight}")
        except Exception as e:
            print(f"❌ Failed to load LSTM weights: {e}")
            
        self.model.eval()

        self.track_skeletons = defaultdict(lambda: deque(maxlen=seq_len))
        self.track_start_times = {}
        self.track_recent_preds = defaultdict(lambda: deque(maxlen=7))

    def _setup_logger(self):
        log_file = settings.data_dir / "logs" / f"realtime_log_{datetime.now().strftime('%Y%m%d')}.csv"
        if not log_file.exists():
            with open(log_file, mode='w', newline='') as f:
                csv.writer(f).writerow(["Timestamp", "Track_ID", "Behavior", "Confidence"])
        return log_file

    def recognize_from_video(self, source, display=True):
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"Lỗi: Không thể mở video {source}")
            return
            
        log_file = self._setup_logger()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            current_time = time.time()
            persons = self.tracker.track_frame(frame)
            current_preds = {}
            current_ids = [p["track_id"] for p in persons]

            for tid in list(self.track_start_times.keys()):
                if tid not in current_ids:
                    del self.track_start_times[tid]
                    if tid in self.track_recent_preds:
                        del self.track_recent_preds[tid]

            for p in persons:
                tid = p["track_id"]
                bbox = p["bbox"]
                
                if tid not in self.track_start_times:
                    self.track_start_times[tid] = current_time
                time_tracked = current_time - self.track_start_times[tid]

                self.track_skeletons[tid].append(p["keypoints"])
                label, conf = "...", 0.0

                if len(self.track_skeletons[tid]) >= 15:
                    x = self.preprocessor(list(self.track_skeletons[tid]))
                    
                    with torch.no_grad():
                        logits = self.model(x)
                        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
                        cls_id = int(np.argmax(probs))  # NumPy giờ đã chạy ngon!
                        raw_conf = float(probs[cls_id])
                        raw_label = self.classes[cls_id]
                        
                    label, conf = self.postprocessor.process(
                        raw_label, raw_conf, self.track_recent_preds[tid], time_tracked
                    )

                current_preds[tid] = (label, conf, bbox)

            for tid, (label, conf, bbox) in current_preds.items():
                if label != "normal" and label != "...":
                    with open(log_file, mode='a', newline='') as f:
                        csv.writer(f).writerow([datetime.now().strftime('%Y-%m-%d %H:%M:%S'), tid, label, f"{conf:.2f}"])

                x1, y1, x2, y2 = map(int, bbox)
                color = (0, 255, 0)
                if label == "fighting": color = (0, 0, 255)
                elif label == "falling": color = (0, 165, 255)
                elif label == "loitering": color = (255, 0, 0)
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"ID {tid} | {label} ({conf:.2f})", (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                kp = self.track_skeletons[tid][-1]
                for px, py, conf_kp in kp:
                    if conf_kp > 0.3:
                        cv2.circle(frame, (int(px), int(py)), 3, color, -1)

            if display:
                cv2.imshow("Realtime Behavior Analytics", frame)
                if cv2.waitKey(1) & 0xFF == 27: break

        cap.release()
        cv2.destroyAllWindows()