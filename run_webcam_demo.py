# run_webcam_demo.py
"""Webcam demo with real detection - chạy riêng cho demo realtime"""

import cv2
import numpy as np
import torch
from collections import defaultdict, deque
from ultralytics import YOLO
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from config.settings import settings
from core.detection.pose_estimator import PoseEstimator
from core.models.lstm_skeleton import SkeletonLSTM

def preprocess_sequence(seq_kp, seq_len=20):
    """Preprocess keypoints sequence"""
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

    # Calculate velocities
    if T >= 2:
        velocity = np.zeros_like(seq_rel)
        velocity[1:] = seq_rel[1:] - seq_rel[:-1]
        velocity[0] = velocity[1]
    else:
        velocity = np.zeros_like(seq_rel)

    # Pad or crop
    if T < seq_len:
        pad_len = seq_len - T
        pad_xy = np.repeat(seq_rel[-1:], pad_len, axis=0)
        pad_vel = np.zeros((pad_len, 17, 2))
        seq_rel = np.concatenate([seq_rel, pad_xy], axis=0)
        velocity = np.concatenate([velocity, pad_vel], axis=0)
    elif T > seq_len:
        seq_rel = seq_rel[-seq_len:]
        velocity = velocity[-seq_len:]

    seq_xy_flat = seq_rel.reshape(seq_len, -1)
    vel_flat = velocity.reshape(seq_len, -1)
    seq_flat = np.concatenate([seq_xy_flat, vel_flat], axis=1)
    
    return torch.from_numpy(seq_flat).float()

def main():
    print("="*50)
    print("WEBCAM REALTIME DEMO")
    print("="*50)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # Load models
    print("Loading models...")
    yolo_model = YOLO(settings.model.yolo_weight)
    pose_model = PoseEstimator(settings.model.pose_weight, device)
    
    # Load LSTM
    classes = settings.classes
    lstm_model = SkeletonLSTM(
        input_size=68,
        hidden_size=256,
        num_layers=2,
        num_classes=len(classes)
    ).to(device)
    lstm_model.load_state_dict(torch.load(settings.model.lstm_weight, map_location=device))
    lstm_model.eval()
    
    print("Models loaded!")
    
    # Open webcam
    print("\nOpening webcam...")
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    if not cap.isOpened():
        print("ERROR: Cannot open webcam!")
        return
    
    print("Webcam ready! Press ESC to quit")
    print("-"*50)
    
    # Tracking data
    track_skeletons = defaultdict(lambda: deque(maxlen=20))
    
    # Stats
    frame_count = 0
    fps = 0
    import time
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        
        frame_count += 1
        current_time = time.time()
        
        # Calculate FPS
        if frame_count % 30 == 0:
            fps = 30 / (current_time - start_time)
            start_time = current_time
        
        # Track persons
        results = yolo_model.track(
            frame, 
            persist=True,
            classes=[0],
            conf=0.4,
            verbose=False
        )
        
        if results and len(results) > 0:
            result = results[0]
            
            # Get detections
            dets = []
            if result.boxes is not None and result.boxes.id is not None:
                ids = result.boxes.id.cpu().numpy()
                boxes = result.boxes.xyxy.cpu().numpy()
                confs = result.boxes.conf.cpu().numpy()
                
                for tid, box, conf in zip(ids, boxes, confs):
                    x1, y1, x2, y2 = box.astype(int)
                    dets.append({
                        "track_id": int(tid),
                        "bbox": [x1, y1, x2, y2],
                        "conf": float(conf)
                    })
            
            # Pose estimation
            persons_pose = pose_model.estimate(frame)
            
            # Process each person
            for d in dets:
                tid = d["track_id"]
                bbox = d["bbox"]
                
                # Find matching pose
                best_kp = None
                best_iou = 0.0
                for p in persons_pose:
                    # Simple IoU calculation
                    pb = p["bbox"]
                    x1 = max(bbox[0], pb[0])
                    y1 = max(bbox[1], pb[1])
                    x2 = min(bbox[2], pb[2])
                    y2 = min(bbox[3], pb[3])
                    inter = max(0, x2-x1) * max(0, y2-y1)
                    a1 = (bbox[2]-bbox[0]) * (bbox[3]-bbox[1])
                    a2 = (pb[2]-pb[0]) * (pb[3]-pb[1])
                    union = a1 + a2 - inter
                    iou = inter / union if union > 0 else 0
                    
                    if iou > best_iou:
                        best_iou = iou
                        best_kp = p["keypoints"]
                
                # Store keypoints
                if best_kp is not None and best_iou > 0.3:
                    track_skeletons[tid].append(best_kp)
                
                # Predict behavior
                label = "..."
                conf = 0.0
                color = (128, 128, 128)
                
                seq_kp = list(track_skeletons[tid])
                if len(seq_kp) >= 10:
                    x = preprocess_sequence(seq_kp).unsqueeze(0).to(device)
                    
                    with torch.no_grad():
                        logits = lstm_model(x)
                        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
                        cls_id = int(np.argmax(probs))
                        conf = float(probs[cls_id])
                        label = classes[cls_id]
                    
                    # Color based on behavior
                    if label == "normal":
                        color = (0, 255, 0)
                    elif label == "fighting":
                        color = (0, 0, 255)
                    elif label == "falling":
                        color = (0, 165, 255)
                    elif label == "loitering":
                        color = (255, 0, 255)
                
                # Draw results
                x1, y1, x2, y2 = bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Label with background
                text = f"ID{tid}: {label}"
                if conf > 0:
                    text += f" ({conf:.2f})"
                
                label_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (x1, y1-25), (x1+label_size[0]+10, y1), color, -1)
                cv2.putText(frame, text, (x1+5, y1-8),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw stats
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Frame: {frame_count}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Add title
        cv2.putText(frame, "ABNORMAL BEHAVIOR DETECTION - REALTIME DEMO", (10, frame.shape[0]-20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Show frame
        cv2.imshow("Webcam - Abnormal Behavior Detection (Press ESC to quit)", frame)
        
        # Check for ESC key
        if cv2.waitKey(1) & 0xFF == 27:
            break
    
    print("\nClosing...")
    cap.release()
    cv2.destroyAllWindows()
    print("Done!")

if __name__ == "__main__":
    main()
