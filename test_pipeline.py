# test_pipeline.py
"""Test script để kiểm tra toàn bộ pipeline"""

import sys
from pathlib import Path
import torch
import cv2
import numpy as np

# Add project root to path
sys.path.append(str(Path(__file__).parent))

def test_imports():
    """Test all imports"""
    print("Testing imports...")
    try:
        from config.settings import settings
        print("✅ Config loaded")
        
        from core.detection.tracker import PersonTracker
        print("✅ Tracker imported")
        
        from core.detection.pose_estimator import PoseEstimator
        print("✅ Pose estimator imported")
        
        from core.models.lstm_skeleton import SkeletonLSTM
        print("✅ LSTM model imported")
        
        from core.inference.pipeline import RealtimeBehaviorRecognizer
        print("✅ Pipeline imported")
        
        from api.services.recognition import RecognitionService
        print("✅ Recognition service imported")
        
        return True
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

def test_model_weights():
    """Test if model weights exist"""
    print("\nTesting model weights...")
    from config.settings import settings
    
    weights = [
        settings.weights_dir / "detection/yolov8s.pt",
        settings.weights_dir / "detection/yolov8s-pose.pt",
        settings.weights_dir / "classification/lstm_best.pth"
    ]
    
    all_exist = True
    for weight_path in weights:
        if weight_path.exists():
            print(f"✅ {weight_path.name} found")
        else:
            print(f"❌ {weight_path.name} NOT found at {weight_path}")
            all_exist = False
    
    return all_exist

def test_device():
    """Test CUDA availability"""
    print("\nTesting device...")
    if torch.cuda.is_available():
        print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("⚠️ CUDA not available, using CPU")
    
    return True

def test_inference_pipeline():
    """Test inference pipeline with dummy video"""
    print("\nTesting inference pipeline...")
    try:
        from config.settings import settings
        from core.inference.pipeline import RealtimeBehaviorRecognizer
        
        # Initialize recognizer
        recognizer = RealtimeBehaviorRecognizer(
            yolo_weight=str(settings.weights_dir / "detection/yolov8s.pt"),
            pose_weight=str(settings.weights_dir / "detection/yolov8s-pose.pt"),
            lstm_weight=str(settings.weights_dir / "classification/lstm_best.pth"),
            classes=settings.classes,
            device="cpu",  # Use CPU for testing
            seq_len=settings.model.seq_len,
            min_frames_for_decision=settings.model.min_frames
        )
        print("✅ Pipeline initialized successfully")
        
        # Create dummy video
        print("Creating dummy video for testing...")
        dummy_video_path = Path("test_video.mp4")
        
        # Create a simple video with moving rectangle
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(dummy_video_path), fourcc, 20.0, (640, 480))
        
        for i in range(60):  # 3 seconds at 20 FPS
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            # Draw a moving person-like rectangle
            x = 100 + i * 5
            y = 200
            cv2.rectangle(frame, (x, y), (x+50, y+100), (0, 255, 0), -1)
            cv2.putText(frame, f"Frame {i}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            out.write(frame)
        
        out.release()
        print(f"✅ Dummy video created: {dummy_video_path}")
        
        # Test processing
        output_path = Path("test_output.mp4")
        print(f"Processing video...")
        recognizer.recognize_from_video(
            source=str(dummy_video_path),
            display=False,
            save_output=str(output_path)
        )
        
        if output_path.exists():
            print(f"✅ Output video created: {output_path}")
            # Clean up
            dummy_video_path.unlink()
            output_path.unlink()
            return True
        else:
            print("❌ Failed to create output video")
            return False
            
    except Exception as e:
        print(f"❌ Pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_api_services():
    """Test API services"""
    print("\nTesting API services...")
    try:
        from api.services.recognition import RecognitionService
        from api.services.storage import StorageManager
        
        # Test storage
        from config.settings import settings
        test_content = b"test data"
        test_path = StorageManager.save_upload("test_id", "test.mp4", test_content)
        
        if test_path.exists():
            print("✅ Storage service working")
            StorageManager.cleanup_upload(test_path)
        else:
            print("❌ Storage service failed")
            
        # Test recognition service init
        RecognitionService.initialize()
        print("✅ Recognition service initialized")
        
        return True
    except Exception as e:
        print(f"❌ API services test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("="*50)
    print("TESTING ABNORMAL BEHAVIOR DETECTION PIPELINE")
    print("="*50)
    
    tests = [
        ("Imports", test_imports),
        ("Model Weights", test_model_weights),
        ("Device", test_device),
        ("API Services", test_api_services),
        ("Inference Pipeline", test_inference_pipeline),
    ]
    
    results = []
    for name, test_func in tests:
        print(f"\n[{name}]")
        result = test_func()
        results.append((name, result))
        print("-"*30)
    
    print("\n" + "="*50)
    print("TEST SUMMARY:")
    print("="*50)
    
    all_passed = True
    for name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{name}: {status}")
        if not result:
            all_passed = False
    
    if all_passed:
        print("\n🎉 All tests passed! System is ready to run.")
        print("\nNext steps:")
        print("1. Run API: cd api && python app.py")
        print("2. Run Web: cd web && streamlit run app.py")
    else:
        print("\n⚠️ Some tests failed. Please fix the issues before running.")
        print("\nCommon fixes:")
        print("1. Install requirements: pip install -r requirements.txt")
        print("2. Download YOLO weights if missing")
        print("3. Check if lstm_best.pth exists in weights/classification/")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
