# src/inference/realtime_pipeline.py
import cv2
import numpy as np
import torch
from collections import defaultdict, deque
from typing import List, Tuple, Dict

from src.detection.tracker import PersonTracker
from src.detection.pose_estimator import PoseEstimator
from src.models.lstm_skeleton import SkeletonLSTM

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

class RealtimeBehaviorRecognizer:
    def __init__(self,
                 yolo_weight: str,
                 pose_weight: str,
                 lstm_weight: str,
                 classes: List[str],
                 device: str = "cuda",
                 seq_len: int = 20,
                 min_frames_for_decision: int = 10):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.classes = classes
        self.seq_len = seq_len
        self.min_frames_for_decision = min_frames_for_decision

        self.tracker = PersonTracker(yolo_weight, self.device)
        self.pose_model = PoseEstimator(pose_weight, self.device)

        self.model = SkeletonLSTM(
            input_size=34,
            hidden_size=128,
            num_layers=2,
            num_classes=len(classes)
        ).to(self.device)
        self.model.load_state_dict(torch.load(lstm_weight, map_location=self.device))
        self.model.eval()

        self.track_skeletons = defaultdict(lambda: deque(maxlen=self.seq_len))

    def _preprocess_sequence(self, seq_kp: List[np.ndarray]) -> torch.Tensor:
        T = len(seq_kp)
        arr = np.zeros((T, 17, 3), dtype=np.float32)
        for i in range(T):
            arr[i] = seq_kp[i]

        seq_xy = arr[:, :, :2]
        center = seq_xy.mean(axis=1, keepdims=True)
        seq_rel = seq_xy - center
        scale = np.linalg.norm(seq_rel, axis=2).max()
        if scale > 0:
            seq_rel /= scale

        if T < self.seq_len:
            pad = np.repeat(seq_rel[-1][None, :, :], self.seq_len - T, axis=0)
            seq_rel = np.concatenate([seq_rel, pad], axis=0)
        elif T > self.seq_len:
            seq_rel = seq_rel[-self.seq_len:]

        seq_flat = seq_rel.reshape(self.seq_len, -1)
        x = torch.from_numpy(seq_flat).float().unsqueeze(0).to(self.device)
        return x

    def recognize_from_video(self, source, display=True, save_output=None):
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open source: {source}")

        writer = None
        if save_output is not None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            fps = cap.get(cv2.CAP_PROP_FPS) or 25
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            writer = cv2.VideoWriter(save_output, fourcc, fps, (w, h))

        track_iter = self.tracker.track_stream(source)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            try:
                dets = next(track_iter)
            except StopIteration:
                break

            persons_pose = self.pose_model.estimate(frame)
            current_preds: Dict[int, Tuple[str, float]] = {}

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
                    self.track_skeletons[tid].append(best_kp)

                seq_kp = list(self.track_skeletons[tid])
                if len(seq_kp) >= self.min_frames_for_decision:
                    x = self._preprocess_sequence(seq_kp)
                    with torch.no_grad():
                        logits = self.model(x)
                        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
                        cls_id = int(np.argmax(probs))
                        conf = float(probs[cls_id])
                        label = self.classes[cls_id]
                    current_preds[tid] = (label, conf)
                else:
                    current_preds[tid] = ("...", 0.0)

            for d in dets:
                tid = d["track_id"]
                x1, y1, x2, y2 = d["bbox"]
                label, conf = current_preds.get(tid, ("", 0.0))

                color = (0, 255, 0)
                if label == "fighting":
                    color = (0, 0, 255)
                elif label == "falling":
                    color = (0, 165, 255)
                elif label == "loitering":
                    color = (255, 0, 0)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                txt = f"ID {tid} | {label}"
                if conf > 0:
                    txt += f" ({conf:.2f})"
                cv2.putText(frame, txt, (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            if display:
                cv2.imshow("Realtime Behavior Recognition", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break

            if writer is not None:
                writer.write(frame)

        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()
