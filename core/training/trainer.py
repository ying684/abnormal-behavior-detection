# core/training/trainer.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Optional, Dict, Any
import json
from datetime import datetime

from config.settings import settings
from core.models.lstm_skeleton import SkeletonLSTM
from core.training.dataset import SkeletonDataset
from core.training.evaluation import Evaluator

class Trainer:
    """Unified trainer for all models"""
    
    def __init__(
        self,
        model_type: str = "lstm",
        experiment_name: Optional[str] = None
    ):
        self.model_type = model_type
        self.experiment_name = experiment_name or f"{model_type}_{datetime.now():%Y%m%d_%H%M%S}"
        self.checkpoint_dir = Path(settings.model.checkpoint_dir) / self.experiment_name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.device = torch.device(settings.device if torch.cuda.is_available() else "cpu")
        self.model = None
        self.train_loader = None
        self.val_loader = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = nn.CrossEntropyLoss()
        
        # Training state
        self.current_epoch = 0
        self.best_val_acc = 0.0
        self.training_history = []
        
    def setup_model(self, **kwargs):
        """Setup model based on type"""
        if self.model_type == "lstm":
            self.model = SkeletonLSTM(
                input_size=kwargs.get('input_size', settings.model.input_size),
                hidden_size=kwargs.get('hidden_size', settings.model.hidden_size),
                num_layers=kwargs.get('num_layers', settings.model.num_layers),
                num_classes=len(settings.classes),
                dropout=kwargs.get('dropout', settings.model.dropout)
            ).to(self.device)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def setup_data(self, data_dir: Optional[Path] = None):
        """Setup data loaders"""
        if data_dir is None:
            data_dir = settings.data_dir / "processed/sequences"
        
        # Create datasets
        train_dataset = SkeletonDataset(
            root_dir=data_dir,
            classes=settings.classes,
            split="train",
            seq_len=settings.model.seq_len,
            augment=settings.training.augment
        )
        
        val_dataset = SkeletonDataset(
            root_dir=data_dir,
            classes=settings.classes,
            split="val",
            seq_len=settings.model.seq_len,
            augment=False
        )
        
        # Create loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=settings.training.batch_size,
            shuffle=True,
            num_workers=settings.training.num_workers
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=settings.training.batch_size,
            shuffle=False,
            num_workers=settings.training.num_workers
        )
    
    def setup_optimizer(self):
        """Setup optimizer and scheduler"""
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=settings.training.learning_rate,
            weight_decay=settings.training.weight_decay
        )
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=5,
            verbose=True
        )
    
    def train_epoch(self) -> Dict[str, float]:
        """Train one epoch"""
        self.model.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item() * data.size(0)
            pred = output.argmax(dim=1)
            total_correct += (pred == target).sum().item()
            total_samples += data.size(0)
        
        return {
            'loss': total_loss / total_samples,
            'accuracy': total_correct / total_samples
        }
    
    def validate(self) -> Dict[str, float]:
        """Validate model"""
        self.model.eval()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.model(data)
                loss = self.criterion(output, target)
                
                total_loss += loss.item() * data.size(0)
                pred = output.argmax(dim=1)
                total_correct += (pred == target).sum().item()
                total_samples += data.size(0)
        
        return {
            'loss': total_loss / total_samples,
            'accuracy': total_correct / total_samples
        }
    
    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_acc': self.best_val_acc,
            'settings': {
                'model_type': self.model_type,
                'classes': settings.classes,
                'model_config': settings.model.__dict__
            }
        }
        
        # Save checkpoint
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{self.current_epoch}.pth"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pth"
            torch.save(self.model.state_dict(), best_path)
            
            # Also save to main weights directory
            main_weight_path = Path(settings.model.lstm_weight)
            torch.save(self.model.state_dict(), main_weight_path)
    
    def load_checkpoint(self, checkpoint_path: Path):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_acc = checkpoint['best_val_acc']
    
    def train(self, num_epochs: Optional[int] = None):
        """Main training loop"""
        if num_epochs is None:
            num_epochs = settings.training.num_epochs
        
        print(f"Starting training: {self.experiment_name}")
        print(f"Device: {self.device}")
        print(f"Model: {self.model_type}")
        print(f"Classes: {settings.classes}")
        print("-" * 50)
        
        early_stopping_counter = 0
        
        for epoch in range(1, num_epochs + 1):
            self.current_epoch = epoch
            
            # Train
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            # Update scheduler
            self.scheduler.step(val_metrics['accuracy'])
            
            # Print progress
            print(f"Epoch {epoch}/{num_epochs}")
            print(f"  Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}")
            print(f"  Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}")
            
            # Save history
            self.training_history.append({
                'epoch': epoch,
                'train': train_metrics,
                'val': val_metrics
            })
            
            # Check best model
            if val_metrics['accuracy'] > self.best_val_acc:
                self.best_val_acc = val_metrics['accuracy']
                self.save_checkpoint(is_best=True)
                print(f"  → New best model! Accuracy: {self.best_val_acc:.4f}")
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
                
            # Early stopping
            if early_stopping_counter >= settings.training.early_stopping_patience:
                print(f"Early stopping triggered after {epoch} epochs")
                break
            
            # Regular checkpoint
            if epoch % 5 == 0:
                self.save_checkpoint(is_best=False)
        
        # Save training history
        history_path = self.checkpoint_dir / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        
        print("-" * 50)
        print(f"Training completed! Best validation accuracy: {self.best_val_acc:.4f}")
        print(f"Results saved to: {self.checkpoint_dir}")
        
        return self.training_history
