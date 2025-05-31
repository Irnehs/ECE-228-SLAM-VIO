import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F


class Decoder(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=128, output_dim=7, num_layers=1, dropout=0.3):
        super().__init__()
        self.rnn_cell = nn.LSTMCell(input_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, context_seq, prediction_len=None):
        # CHECK THIS factor in prediction_len for variable output length
        summary = context_seq[:, -1, :]  # [B, D]
        B = summary.shape[0]

        h = summary
        c = torch.zeros(B, h.size(1), device=h.device)
        input_step = torch.zeros(B, self.rnn_cell.input_size, device=device)

        preds = []
        for _ in range(prediction_len):
            h, c = self.rnn_cell(h, (h, c))
            h = self.dropout(h)
            pred = self.out(h)  # [B, 7]
            preds.append(pred.unsqueeze(1))

        return torch.cat(preds, dim=1)  # [B, prediction_len, 7]