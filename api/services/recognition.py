# api/services/recognition.py
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch
from typing import Optional

from config.settings import settings
from core.inference.pipeline import RealtimeBehaviorRecognizer

class RecognitionService:
    _instance: Optional[RealtimeBehaviorRecognizer] = None
    
    @classmethod
    def initialize(cls):
        if cls._instance is None:
            print("Initializing recognition model...")
            
            device = settings.device if torch.cuda.is_available() else "cpu"
            print(f"Using device: {device}")
            
            cls._instance = RealtimeBehaviorRecognizer(
                yolo_weight=settings.model.yolo_weight,
                pose_weight=settings.model.pose_weight,
                lstm_weight=settings.model.lstm_weight,
                classes=settings.classes,
                device=device,
                seq_len=settings.model.seq_len,
                min_frames_for_decision=settings.model.min_frames
            )
            print("Model initialized successfully")
    
    @classmethod
    def get_instance(cls) -> RealtimeBehaviorRecognizer:
        if cls._instance is None:
            cls.initialize()
        return cls._instance
    
    @classmethod
    def process_video(cls, input_path: Path, output_id: str) -> Path:
        output_path = settings.data_dir / "outputs" / f"output_{output_id}.mp4"
        
        recognizer = cls.get_instance()
        recognizer.recognize_from_video(
            source=str(input_path),
            display=False,
            save_output=str(output_path)
        )
        
        return output_path
