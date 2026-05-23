# src/detection/pose_estimator.py
from ultralytics import YOLO
import numpy as np
from typing import List, Dict

class PoseEstimator:
    def __init__(self, weight_path: str, device: str = "cuda"):
        self.model = YOLO(weight_path)
        self.model.to(device)
        self.device = device

    def estimate(self, frame) -> List[Dict]:
        results = self.model.predict(
            frame,
            device=self.device,
            verbose=False,
            conf=0.5
        )[0]

        persons = []
        if results.keypoints is None or results.boxes is None:
            return persons

        kps_xy = results.keypoints.xy.cpu().numpy()
        kps_conf = results.keypoints.conf.cpu().numpy()
        boxes = results.boxes.xyxy.cpu().numpy()

        for i in range(kps_xy.shape[0]):
            x1, y1, x2, y2 = boxes[i].astype(int)
            kp_xy = kps_xy[i]
            kp_c = kps_conf[i][:, None]
            kps = np.concatenate([kp_xy, kp_c], axis=1)

            persons.append({
                "bbox": [x1, y1, x2, y2],
                "keypoints": kps
            })

        return persons
