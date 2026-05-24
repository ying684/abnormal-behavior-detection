# core/models/lstm_skeleton.py
import torch
import torch.nn as nn

class SkeletonLSTM(nn.Module):
    def __init__(self,
                 input_size: int = 68,  # 17 keypoints × 4 features
                 hidden_size: int = 256,
                 num_layers: int = 2,
                 num_classes: int = 4,
                 dropout: float = 0.3):
        super().__init__()
        
        # Layer norm đầu vào
        self.layer_norm = nn.LayerNorm(input_size)
        
        # LSTM bidirectional
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )
        
        # FC layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 128),  # 512 -> 128
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # x: (B,T,68)
        x = self.layer_norm(x)
        out, _ = self.lstm(x)    # (B,T,512)
        out = out.mean(dim=1)    # (B,512)
        logits = self.fc(out)    # (B,4)
        return logits
