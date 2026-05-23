# src/train/train_skeleton.py
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.optim import AdamW
from pathlib import Path

from src.data.dataset_skeleton import SkeletonDataset
from src.models.lstm_skeleton import SkeletonLSTM

def train_skeleton_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    classes = ["normal", "fighting", "falling", "loitering"]

    train_ds = SkeletonDataset(
        root_dir="data/processed/sequences",
        classes=classes,
        split="train",
        seq_len=20
    )
    val_ds = SkeletonDataset(
        root_dir="data/processed/sequences",
        classes=classes,
        split="val",
        seq_len=20
    )

    train_loader = DataLoader(train_ds, batch_size=48, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=48, shuffle=False, num_workers=2)

    model = SkeletonLSTM(
        input_size=34,
        hidden_size=128,
        num_layers=2,
        num_classes=len(classes),
        dropout=0.2  # Dropout only works with num_layers >= 2
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=2e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=12, eta_min=1e-5)

    best_val_acc = 0.0
    weights_dir = Path("weights")
    weights_dir.mkdir(parents=True, exist_ok=True)

    num_epochs = 12  # Balance: nhanh + đủ để hội tụ

    for epoch in range(1, num_epochs+1):
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
        scheduler.step()  # Update learning rate
        print(f"Epoch {epoch}/{num_epochs} | train loss {train_loss:.4f} acc {train_acc:.3f} | val acc {val_acc:.3f} | lr {optimizer.param_groups[0]['lr']:.2e}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            out_path = weights_dir / "skeleton_lstm_best.pth"
            torch.save(model.state_dict(), out_path)
            print(f"  -> New best model saved to {out_path}, acc={best_val_acc:.3f}")

if __name__ == "__main__":
    train_skeleton_model()
