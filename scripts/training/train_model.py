# scripts/training/train_model.py
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.optim import AdamW

from config.settings import settings
from core.training.dataset import SkeletonDataset
from core.models.lstm_skeleton import SkeletonLSTM

def train():
    device = settings.device if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Datasets
    train_ds = SkeletonDataset(
        root_dir=settings.data_dir / "processed/sequences",
        classes=settings.classes,
        split="train",
        seq_len=20
    )
    val_ds = SkeletonDataset(
        root_dir=settings.data_dir / "processed/sequences",
        classes=settings.classes,
        split="val",
        seq_len=20
    )
    
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=2)
    
    # Model
    model = SkeletonLSTM(
        input_size=68,
        hidden_size=256,
        num_layers=2,
        num_classes=len(settings.classes)
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    best_val_acc = 0.0
    weights_dir = settings.weights_dir / "classification"
    weights_dir.mkdir(parents=True, exist_ok=True)
    
    num_epochs = 15
    
    for epoch in range(1, num_epochs+1):
        # Train
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * x.size(0)
            preds = logits.argmax(dim=1)
            total_correct += (preds == y).sum().item()
            total_samples += x.size(0)
        
        train_loss = total_loss / total_samples
        train_acc = total_correct / total_samples
        
        # Validation
        model.eval()
        val_correct = 0
        val_samples = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                preds = logits.argmax(dim=1)
                val_correct += (preds == y).sum().item()
                val_samples += x.size(0)
        
        val_acc = val_correct / val_samples if val_samples > 0 else 0.0
        print(f"Epoch {epoch}/{num_epochs} | train loss {train_loss:.4f} acc {train_acc:.3f} | val acc {val_acc:.3f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            out_path = weights_dir / "lstm_best.pth"
            torch.save(model.state_dict(), out_path)
            print(f"  → Saved best model, acc={best_val_acc:.3f}")

if __name__ == "__main__":
    train()
