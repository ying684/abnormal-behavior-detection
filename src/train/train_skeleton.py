# src/train/train_skeleton.py
import torch
import numpy as np
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pathlib import Path

from src.data.dataset_skeleton import SkeletonDataset
from src.models.lstm_skeleton import SkeletonLSTM

def train_skeleton_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    classes = ["normal", "fighting", "falling", "loitering"]
    seq_len = 30 
    
    # BẬT augment=True cho tập train
    train_ds = SkeletonDataset(root_dir="data/processed/sequences", classes=classes, split="train", seq_len=seq_len, augment=True)
    val_ds = SkeletonDataset(root_dir="data/processed/sequences", classes=classes, split="val", seq_len=seq_len, augment=False)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=2)

    # Đảm bảo input_size = 68 để khớp với Dataset mới
    model = SkeletonLSTM(input_size=68, hidden_size=256, num_layers=2, num_classes=len(classes)).to(device)

    labels = [sample[1] for sample in train_ds.samples]
    class_counts = np.bincount(labels, minlength=len(classes))
    class_weights = len(labels) / (len(classes) * np.maximum(class_counts, 1))
    class_weights_tensor = torch.FloatTensor(class_weights).to(device)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

    best_val_acc = 0.0
    weights_dir = Path("weights")
    weights_dir.mkdir(parents=True, exist_ok=True)
    num_epochs = 40 

    for epoch in range(1, num_epochs+1):
        model.train()
        total_loss, total_correct, total_samples_epoch = 0.0, 0, 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # Chống nổ gradient
            optimizer.step()

            total_loss += loss.item() * x.size(0)
            total_correct += (logits.argmax(dim=1) == y).sum().item()
            total_samples_epoch += x.size(0)

        train_loss = total_loss / total_samples_epoch
        train_acc = total_correct / total_samples_epoch

        model.eval()
        val_correct, val_samples = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                val_correct += (logits.argmax(dim=1) == y).sum().item()
                val_samples += x.size(0)

        val_acc = val_correct / val_samples if val_samples > 0 else 0.0
        scheduler.step(val_acc)

        print(f"Epoch {epoch:02d}/{num_epochs} | train loss {train_loss:.4f} acc {train_acc:.3f} | val acc {val_acc:.3f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            out_path = weights_dir / "skeleton_lstm_best.pth"
            torch.save(model.state_dict(), out_path)
            print(f"  -> New best model saved to {out_path}, acc={best_val_acc:.3f}")

if __name__ == "__main__":
    train_skeleton_model()