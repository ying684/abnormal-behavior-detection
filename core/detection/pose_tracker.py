# core/detection/pose_tracker.py

from ultralytics import YOLO
import numpy as np

class PoseTracker:
    def __init__(self, weight_path: str, device: str = "cuda"):
        # Chỉ load 1 model pose duy nhất
        self.model = YOLO(weight_path)
        self.model.to(device)
        self.device = device

    def track_frame(self, frame, conf_thres: float = 0.4):
        """
        Thực hiện inference và tracking trên 1 frame duy nhất.
        Trả về danh sách các người (persons) kèm ID và Keypoints.
        """
        results = self.model.track(
            frame,
            conf=conf_thres,
            device=self.device,
            classes=[0],        # Chỉ tracking người
            persist=True,       # Bắt buộc để giữ ID giữa các frame
            verbose=False,
            tracker="botsort.yaml" # Dùng BoT-SORT ổn định hơn DeepSORT
        )[0]

        persons = []
        
        # Nếu không có boxes, không có keypoints, hoặc chưa gán được ID
        if results.boxes is None or results.keypoints is None or results.boxes.id is None:
            return persons

        ids = results.boxes.id.cpu().numpy().astype(int)
        boxes = results.boxes.xyxy.cpu().numpy().astype(int)
        confs = results.boxes.conf.cpu().numpy()
        
        kps_xy = results.keypoints.xy.cpu().numpy()       # (N, 17, 2)
        kps_conf = results.keypoints.conf.cpu().numpy()   # (N, 17)

        for i in range(len(ids)):
            x1, y1, x2, y2 = boxes[i]
            kp_xy = kps_xy[i]            # (17, 2)
            kp_c = kps_conf[i][:, None]  # (17, 1)
            kps = np.concatenate([kp_xy, kp_c], axis=1) # (17, 3)

            persons.append({
                "track_id": ids[i],
                "bbox": [x1, y1, x2, y2],
                "bbox_conf": float(confs[i]),
                "keypoints": kps
            })

        return persons