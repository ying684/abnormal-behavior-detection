# scripts/training/train_model.py

import sys
from pathlib import Path

sys.path.append(
    str(Path(__file__).parent.parent.parent)
)

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from config.settings import settings
from core.training.dataset import SkeletonDataset
from core.models.lstm_skeleton import SkeletonLSTM
from core.training.evaluation import Evaluator

def train():

    device = settings.device if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # CHỈ DÙNG 3 CLASS NÀY CHO LSTM
    target_classes = ["normal", "fighting", "falling"]

    # --------------------------
    # DATASET
    # --------------------------
    train_ds = SkeletonDataset(
        root_dir=settings.data_dir / "processed/sequences",
        classes=target_classes,
        split="train",
        seq_len=30,
        augment=True
    )

    val_ds = SkeletonDataset(
        root_dir=settings.data_dir / "processed/sequences",
        classes=target_classes,
        split="val",
        seq_len=30,
        augment=False
    )

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=2)

    # --------------------------
    # MODEL (Input 68, Output 3)
    # --------------------------
    model = SkeletonLSTM(
        input_size=68, 
        hidden_size=128,
        num_layers=2,
        num_classes=len(target_classes)
    ).to(device)

    # Dựa vào dataset: Normal(130), Fighting(147), Falling(108)
    # Tăng nhẹ trọng số cho Falling vì mẫu ít hơn một chút
    weights = torch.tensor([1.0, 2.5, 3.0], dtype=torch.float32).to(device)

    criterion = nn.CrossEntropyLoss(weight=weights)

    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=8
    )

    # --------------------------
    # SAVE DIR
    # --------------------------
    weights_dir = settings.weights_dir / "classification"
    weights_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------
    # TRAINING CONFIG
    # --------------------------
    num_epochs = 50
    best_val_acc = 0.0
    best_cm = None

    early_stop_patience = 15
    early_stop_counter = 0

    # --------------------------
    # TRAIN LOOP
    # --------------------------
    for epoch in range(1, num_epochs + 1):
        model.train()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        # ======================
        # TRAIN
        # ======================
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item() * x.size(0)
            preds = logits.argmax(dim=1)
            total_correct += (preds == y).sum().item()
            total_samples += x.size(0)

        train_loss = total_loss / total_samples
        train_acc = total_correct / total_samples

        # ======================
        # VALIDATION
        # ======================
        model.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                preds = logits.argmax(dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(y.cpu().numpy())

        metrics = Evaluator.evaluate(all_targets, all_preds)
        val_acc = metrics["accuracy"]
        val_f1 = metrics["f1"]

        scheduler.step(val_acc)
        current_lr = optimizer.param_groups[0]["lr"]

        print(f"Epoch {epoch:02d} | LR {current_lr:.6f} | Train Loss {train_loss:.4f} | Train Acc {train_acc:.4f} | Val Acc {val_acc:.4f} | Val F1 {val_f1:.4f}")

        # ======================
        # SAVE BEST MODEL
        # ======================
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_cm = metrics["confusion_matrix"]

            # LƯU Ý KỸ: Lưu trực tiếp state_dict() để dễ load ở pipeline.py
            torch.save(
                model.state_dict(),
                weights_dir / "lstm_best.pth"
            )

            print(f"  → Saved best model (Acc={best_val_acc:.4f})")
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        # ======================
        # EARLY STOPPING
        # ======================
        if early_stop_counter >= early_stop_patience:
            print(f"\nEarly stopping at epoch {epoch}")
            break

    # --------------------------
    # FINAL REPORT
    # --------------------------
    print("\n" + "=" * 50)
    print(f"Best Validation Accuracy: {best_val_acc:.4f}")
    print("\nBest Confusion Matrix:")
    if best_cm is not None:
        print(best_cm)
    print("=" * 50)

if __name__ == "__main__":
    train()