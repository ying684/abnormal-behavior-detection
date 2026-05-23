# main_realtime_demo.py
import sys
from pathlib import Path

# Add project root to path for imports (works both locally and in Colab)
sys.path.insert(0, str(Path(__file__).parent))

from src.inference.realtime_pipeline import RealtimeBehaviorRecognizer

def main():
    yolo_weight = "weights/yolov8s.pt"
    pose_weight = "weights/yolov8s-pose.pt"
    lstm_weight = "weights/skeleton_lstm_best.pth"

    classes = ["normal", "fighting", "falling", "loitering"]

    recognizer = RealtimeBehaviorRecognizer(
        yolo_weight=yolo_weight,
        pose_weight=pose_weight,
        lstm_weight=lstm_weight,
        classes=classes,
        device="cuda",             # fallback sang CPU nếu không có
        seq_len=20,
        min_frames_for_decision=10
    )

    # 0 = webcam; hoặc path đến file .mp4; hoặc rtsp url
    source = 0

    recognizer.recognize_from_video(
        source=source,
        display=True,
        save_output=None
    )

if __name__ == "__main__":
    main()
