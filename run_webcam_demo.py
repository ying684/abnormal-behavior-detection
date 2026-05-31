# run_webcam_demo.py
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from config.settings import settings
from core.inference.pipeline import RealtimeBehaviorRecognizer

def main():
    print("="*50)
    print("🚀 KÍCH HOẠT HỆ THỐNG CAMERA GIÁM SÁT REAL-TIME")
    print("="*50)

    try:
        recognizer = RealtimeBehaviorRecognizer(
            pose_weight=str(settings.model.pose_weight),
            lstm_weight=str(settings.model.lstm_weight),
            # <-- ĐÃ FIX: Chỉ truyền đúng 3 class để model không bị "ngáo"
            classes=["normal", "fighting", "falling"], 
            device=settings.device,
            seq_len=settings.model.seq_len,
            loitering_threshold_sec=10.0
        )
        print("✅ Đã khởi tạo AI thành công!")
    except Exception as e:
        print(f"❌ Lỗi khởi tạo hệ thống: {e}")
        return

    print("\n🎥 Đang mở Camera... (Bấm phím ESC trên cửa sổ video để tắt)")
    recognizer.recognize_from_video(source=0, display=True)

if __name__ == "__main__":
    main()