# config/settings.py
from pathlib import Path
from typing import List, Optional
import os
from dataclasses import dataclass

# Base paths - sử dụng Path để tương thích cross-platform
BASE_DIR = Path(__file__).parent.parent.absolute()
DATA_DIR = BASE_DIR / "data"
WEIGHTS_DIR = BASE_DIR / "weights"
CONFIG_DIR = BASE_DIR / "config"

@dataclass
class ModelConfig:
    """Model configuration"""
    # Detection - sử dụng Path objects
    yolo_weight: Path = WEIGHTS_DIR / "detection" / "yolov8s.pt"
    pose_weight: Path = WEIGHTS_DIR / "detection" / "yolov8s-pose.pt"
    
    # Classification
    lstm_weight: Path = WEIGHTS_DIR / "classification" / "lstm_best.pth"
    checkpoint_dir: Path = WEIGHTS_DIR / "classification" / "checkpoints"
    
    # Model parameters
    input_size: int = 68
    hidden_size: int = 128
    num_layers: int = 2
    num_classes: int = 3
    dropout: float = 0.3
    
    # Inference
    seq_len: int = 20
    min_frames: int = 10
    confidence_threshold: float = 0.4
    iou_threshold: float = 0.3
    
    def __post_init__(self):
        # Convert to string for compatibility
        self.yolo_weight = str(self.yolo_weight)
        self.pose_weight = str(self.pose_weight)
        self.lstm_weight = str(self.lstm_weight)
        self.checkpoint_dir = str(self.checkpoint_dir)

@dataclass
class TrainingConfig:
    """Training configuration"""
    batch_size: int = 32
    num_workers: int = 2  # Giảm xuống 2 cho Windows
    train_split: float = 0.8
    val_split: float = 0.1
    num_epochs: int = 50
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    early_stopping_patience: int = 10
    augment: bool = True
    augment_prob: float = 0.5

@dataclass
class APIConfig:
    """API configuration"""
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = True
    max_upload_size: int = 200 * 1024 * 1024  # 200MB

@dataclass
class Settings:
    """Global settings"""
    # Paths
    base_dir: Path = BASE_DIR
    data_dir: Path = DATA_DIR
    weights_dir: Path = WEIGHTS_DIR
    
    # Configurations
    model: ModelConfig = None
    training: TrainingConfig = None
    api: APIConfig = None
    
    # Behavior classes
    classes: List[str] = None
    
    # Device
    device: str = "cuda"
    
    def __init__(self):
        self.model = ModelConfig()
        self.training = TrainingConfig()
        self.api = APIConfig()
        
        if self.classes is None:
            self.classes = ["normal", "fighting", "falling", "loitering"]
        
        # Create directories
        self.data_dir.mkdir(exist_ok=True)
        self.weights_dir.mkdir(exist_ok=True)
        (self.data_dir / "raw").mkdir(exist_ok=True, parents=True)
        (self.data_dir / "processed").mkdir(exist_ok=True, parents=True)
        (self.data_dir / "cache").mkdir(exist_ok=True, parents=True)
        (self.data_dir / "outputs").mkdir(exist_ok=True, parents=True)
        (self.data_dir / "logs").mkdir(exist_ok=True, parents=True)

# Global settings instance
settings = Settings()
