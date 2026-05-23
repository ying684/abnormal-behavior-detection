# src/models/lstm_skeleton.py
import torch
import torch.nn as nn

class SkeletonLSTM(nn.Module):
    def __init__(self,
                 input_size: int = 34,
                 hidden_size: int = 128,
                 num_layers: int = 2,
                 num_classes: int = 4,
                 dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        out, _ = self.lstm(x)    # (B,T,2H)
        out = out.mean(dim=1)    # (B,2H)
        logits = self.fc(out)    # (B,C)
        return logits
