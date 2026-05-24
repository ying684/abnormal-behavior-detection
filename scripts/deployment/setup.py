# scripts/deployment/setup.py
import os
import sys
import shutil
import subprocess
from pathlib import Path

def setup():
    print("="*50)
    print("Setting up Abnormal Behavior Detection System...")
    print("="*50)
    
    # Check Python
    print(f"Python version: {sys.version}")
    print(f"Python path: {sys.executable}")
    
    # Create directories
    print("\n📁 Creating directories...")
    directories = [
        "weights/detection",
        "weights/classification", 
        "weights/classification/checkpoints",
        "data/raw",
        "data/processed",
        "data/cache",
        "data/outputs",
        "data/logs"
    ]
    
    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"  ✓ {dir_path}")
    
    # Install requirements
    print("\n📦 Installing requirements...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("  ✅ Requirements installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"  ❌ Failed to install requirements: {e}")
        print("  Please run manually: pip install -r requirements.txt")
    
    # Download YOLO weights
    print("\n🎯 Checking YOLO weights...")
    
    # YOLOv8
    yolo_path = Path("weights/detection/yolov8s.pt")
    if not yolo_path.exists():
        print("  Downloading YOLOv8 weights...")
        try:
            from ultralytics import YOLO
            model = YOLO('yolov8s.pt')
            if Path('yolov8s.pt').exists():
                shutil.move('yolov8s.pt', str(yolo_path))
                print(f"  ✅ YOLOv8 weights saved to {yolo_path}")
        except Exception as e:
            print(f"  ❌ Failed to download YOLOv8: {e}")
    else:
        print(f"  ✅ YOLOv8 weights found at {yolo_path}")
    
    # YOLOv8-Pose
    pose_path = Path("weights/detection/yolov8s-pose.pt")
    if not pose_path.exists():
        print("  Downloading YOLOv8-Pose weights...")
        try:
            from ultralytics import YOLO
            model = YOLO('yolov8s-pose.pt')
            if Path('yolov8s-pose.pt').exists():
                shutil.move('yolov8s-pose.pt', str(pose_path))
                print(f"  ✅ YOLOv8-Pose weights saved to {pose_path}")
        except Exception as e:
            print(f"  ❌ Failed to download YOLOv8-Pose: {e}")
    else:
        print(f"  ✅ YOLOv8-Pose weights found at {pose_path}")
    
    # Check LSTM weights
    print("\n🧠 Checking LSTM weights...")
    lstm_path = Path("weights/classification/lstm_best.pth")
    if not lstm_path.exists():
        # Try to find in old location
        old_path = Path("weights/skeleton_lstm_best.pth")
        if old_path.exists():
            shutil.copy(str(old_path), str(lstm_path))
            print(f"  ✅ Copied LSTM weights from old location")
        else:
            print(f"  ⚠️ LSTM weights not found at {lstm_path}")
            print("  Please copy your trained model to: weights/classification/lstm_best.pth")
    else:
        print(f"  ✅ LSTM weights found at {lstm_path}")
    
    # Create .env file
    print("\n⚙️ Checking configuration...")
    env_path = Path(".env")
    env_example_path = Path(".env.example")
    
    if not env_path.exists() and env_example_path.exists():
        shutil.copy(str(env_example_path), str(env_path))
        print("  ✅ Created .env from .env.example")
    elif env_path.exists():
        print("  ✅ .env file exists")
    else:
        print("  ⚠️ No .env file found, using default settings")
    
    # Final check
    print("\n" + "="*50)
    print("Setup Summary:")
    print("="*50)
    
    all_good = True
    
    # Check critical files
    critical_files = [
        ("YOLOv8", "weights/detection/yolov8s.pt"),
        ("YOLOv8-Pose", "weights/detection/yolov8s-pose.pt"),
        ("LSTM Model", "weights/classification/lstm_best.pth"),
        ("Config", "config/settings.py"),
        ("API App", "api/app.py"),
        ("Web App", "web/app.py")
    ]
    
    for name, path in critical_files:
        if Path(path).exists():
            print(f"✅ {name}: Found")
        else:
            print(f"❌ {name}: Not found at {path}")
            all_good = False
    
    print("\n" + "="*50)
    
    if all_good:
        print("✅ Setup completed successfully!")
        print("\nNext steps:")
        print("1. Test pipeline: python test_pipeline.py")
        print("2. Start API: cd api && python app.py")
        print("3. Start Web: cd web && streamlit run app.py")
    else:
        print("⚠️ Setup completed with warnings.")
        print("Please check missing files above.")
    
    return all_good

if __name__ == "__main__":
    success = setup()
    sys.exit(0 if success else 1)
