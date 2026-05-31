# core/models/lstm_skeleton.py

import torch
import torch.nn as nn

class SkeletonLSTM(nn.Module):
    def __init__(
        self,
        input_size=68,
        hidden_size=64,      # GIẢM TỪ 256 XUỐNG 64
        num_layers=1,        # GIẢM TỪ 2 XUỐNG 1
        num_classes=3,
        dropout=0.5          # TĂNG DROPOUT LÊN 0.5 (Chống học vẹt)
    ):
        super().__init__()
        
        self.layer_norm = nn.LayerNorm(input_size)
        
        # Conv1D cũng ép nhỏ lại
        self.conv = nn.Conv1d(
            in_channels=input_size, 
            out_channels=64, 
            kernel_size=3, 
            padding=1
        )
        self.relu = nn.ReLU()

        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )

        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.layer_norm(x)
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = self.relu(x)
        x = x.permute(0, 2, 1)

        lstm_out, _ = self.lstm(x)

        attention_scores = self.attention(lstm_out)
        attention_weights = torch.softmax(attention_scores, dim=1)
        context = torch.sum(lstm_out * attention_weights, dim=1)

        logits = self.fc(context)
        return logits