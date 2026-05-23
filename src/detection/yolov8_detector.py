# src/detection/yolov8_detector.py
from ultralytics import YOLO
from typing import List, Dict

class PersonDetector:
    def __init__(self, weight_path: str, device: str = "cuda"):
        self.model = YOLO(weight_path)
        self.model.to(device)
        self.device = device

    def detect_persons(self, frame, conf_thres: float = 0.4) -> List[Dict]:
        results = self.model.predict(
            frame,
            conf=conf_thres,
            device=self.device,
            verbose=False
        )[0]

        detections = []
        if results.boxes is None:
            return detections

        for box in results.boxes:
            cls_id = int(box.cls[0])
            if cls_id != 0:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            detections.append({
                "bbox": [x1, y1, x2, y2],
                "conf": conf
            })

        return detections
