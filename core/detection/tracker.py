# src/detection/tracker.py
from ultralytics import YOLO
from typing import List, Dict, Union

class PersonTracker:
    def __init__(self, weight_path: str, device: str = "cuda"):
        self.model = YOLO(weight_path)
        self.model.to(device)
        self.device = device

    def track_stream(self, source: Union[str, int], conf_thres: float = 0.4):
        """
        Generator:
            Yield mỗi frame:
            [
              { 'track_id': int, 'bbox': [x1,y1,x2,y2], 'conf': float }
            ]
        """
        track_gen = self.model.track(
            source=source,
            conf=conf_thres,
            device=self.device,
            classes=[0],  # person only
            stream=True,
            persist=True,
            verbose=False
        )

        for result in track_gen:
            frame_dets = []
            if result.boxes is None or result.boxes.id is None:
                yield frame_dets
                continue

            ids = result.boxes.id.cpu().tolist()
            boxes = result.boxes.xyxy.cpu().tolist()
            confs = result.boxes.conf.cpu().tolist()

            for tid, box, conf in zip(ids, boxes, confs):
                x1, y1, x2, y2 = map(int, box)
                frame_dets.append({
                    "track_id": int(tid),
                    "bbox": [x1, y1, x2, y2],
                    "conf": float(conf)
                })
            yield frame_dets
