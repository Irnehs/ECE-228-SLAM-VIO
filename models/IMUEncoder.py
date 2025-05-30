import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
import pandas as pd


class IMUEncoder(nn.Module):
    def __init__(
        self, input_dim=6, hidden_dim=128, output_dim=6, dropout=0.3, window_size=10
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.window_size = window_size
        self.rnn = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.norm = nn.LayerNorm(
            self.hidden_dim * 2
        )  # For bidirectional, we double the hidden dimension
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        B, T, D = x.shape
        assert (
            T % self.window_size == 0
        ), f"Input sequence length {T} must be a multiple of window size {self.window_size}"

        N = T // self.window_size  # Number of windows

        x = x.view(B, N, self.window_size, D)  # Reshape to (B, N, window_size, D)
        x = x.view(B * N, self.window_size, D)  # (B*N, window_size, D)

        _, (hn, _) = self.rnn(x)  # (2, B*N, hidden_dim)
        h = torch.cat((hn[-2], hn[-1]), dim=1)  # (B*N, hidden_dim * 2)

        h = self.norm(h)
        h = self.dropout(h)
        out = self.linear(h)  # (B*N, output_dim)
        out = out.view(B, N, -1)  # (B, N, output_dim)

        return out
