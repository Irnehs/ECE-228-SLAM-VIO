import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
import pandas as pd

class IMUEncoder(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=128, output_dim=7):
        super().__init__()
        self.rnn = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.hidden_size = hidden_dim * 2  # Because bidirectional
        self.norm = nn.LayerNorm(self.hidden_size)
        self.dropout = nn.Dropout(0.3)
        self.linear = nn.Linear(self.hidden_size, output_dim)  # Maps to final 7D output

    def forward(self, x):
        _, (hn, _) = self.rnn(x)
        h = torch.cat((hn[-2], hn[-1]), dim=1)  # shape: (batch, hidden_dim * 2)
        h = self.norm(h)
        h = self.dropout(h)
        out = self.linear(h)  # shape: (batch, 7)
        return out