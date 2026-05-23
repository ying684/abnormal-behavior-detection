# main_realtime_simple.py
from src.inference.realtime_simple import SimpleBehaviorRecognizer

def main():
    pose_weight = "weights/yolov8s-pose.pt"
    lstm_weight = "weights/skeleton_lstm_best.pth"
    classes = ["normal", "fighting", "falling", "loitering"]

    recognizer = SimpleBehaviorRecognizer(
        pose_weight=pose_weight,
        lstm_weight=lstm_weight,
        classes=classes,
        device="cuda",
        seq_len=15,
        min_frames_for_decision=10
    )

    source = 0  # webcam
    recognizer.recognize_from_video(
        source=source,
        display=True,
        save_output=None
    )

if __name__ == "__main__":
    main()
