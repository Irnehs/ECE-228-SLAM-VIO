import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F


class Decoder(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=128, output_dim=7, dropout=0.3):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, context_seq, prediction_len=None):
        # TODO factor in prediction_len for variable output length
        summary = context_seq[:, -1, :]  # [B, F] - Take the last time step as summary

        pred = self.mlp(summary)  # [B, 7]
        return pred.unsqueeze(1)  # [B, 1, 7]
