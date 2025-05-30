import torch.nn as nn
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
        # CHECK THIS factor in prediction_len for variable output length
        summary = context_seq[:, -1, :]  # [B, input_dim], last hidden state summary embedded
        summary_repeated = summary.unsqueeze(1).repeat(1, prediction_len, 1)  #im not sure about this but i just created another to predict k 
        B, L, D = summary_repeated.shape #flatten
        flat = summary_repeated.view(B * L, D)  # 
        pred_flat = self.mlp(flat)  # [B * L, output_dim]
        pred = pred_flat.view(B, prediction_len, -1)  # [B, L, output_dim]
        return pred  # [B, prediction_len, output_dim]???? I'm not squre about this