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
        """
        Input:
            frame: BGR image
        Output:
            list:
            {
              'bbox': [x1,y1,x2,y2],
              'keypoints': np.array (17,3) [x,y,conf]
            }
        """
        results = self.model.predict(
            frame,
            device=self.device,
            verbose=False
        )[0]

        persons = []
        if results.keypoints is None or results.boxes is None:
            return persons

        kps_xy = results.keypoints.xy.cpu().numpy()       # (N,17,2)
        kps_conf = results.keypoints.conf.cpu().numpy()   # (N,17)
        boxes = results.boxes.xyxy.cpu().numpy()          # (N,4)

        for i in range(kps_xy.shape[0]):
            x1, y1, x2, y2 = boxes[i].astype(int)
            kp_xy = kps_xy[i]           # (17,2)
            kp_c = kps_conf[i][:, None] # (17,1)
            kps = np.concatenate([kp_xy, kp_c], axis=1)

            persons.append({
                "bbox": [x1, y1, x2, y2],
                "keypoints": kps
            })

        return persons
