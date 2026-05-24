# api/services/storage.py
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import cv2
from typing import Dict, Any
import os

from config.settings import settings

class StorageManager:
    
    @staticmethod
    def save_upload(upload_id: str, filename: str, contents: bytes) -> Path:
        upload_dir = settings.data_dir / "cache"
        upload_dir.mkdir(exist_ok=True)
        
        safe_filename = f"{upload_id}_{Path(filename).name}"
        file_path = upload_dir / safe_filename
        
        with open(file_path, "wb") as f:
            f.write(contents)
        
        return file_path
    
    @staticmethod
    def cleanup_upload(file_path: Path):
        try:
            if file_path.exists():
                os.unlink(file_path)
        except Exception as e:
            print(f"Cleanup error: {e}")
    
    @staticmethod
    def get_video_info(video_path: Path) -> Dict[str, Any]:
        cap = cv2.VideoCapture(str(video_path))
        
        info = {
            "fps": round(cap.get(cv2.CAP_PROP_FPS), 2),
            "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        }
        
        info["duration"] = round(info["frames"] / info["fps"], 2) if info["fps"] > 0 else 0
        info["resolution"] = f"{info['width']}x{info['height']}"
        
        cap.release()
        
        return info
