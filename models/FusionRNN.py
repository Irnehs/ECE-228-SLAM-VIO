import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F


class FusionRNN(nn.Module):
    def __init__(
        self,
        input_dim=519,
        output_dim=7,
        hidden_dim=128,
        bidirectional=False,
        dropout=0.3,
    ):
        super(FusionRNN, self).__init__()
        self.bidirectional = bidirectional
        self.hidden_size = hidden_dim * 2 if bidirectional else hidden_dim
        self.output_dim = output_dim
        self.dropout_rate = dropout

        self.rnn = nn.LSTM(
            input_dim, hidden_dim, batch_first=True, bidirectional=self.bidirectional
        )
        self.norm = nn.LayerNorm(self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_rate)
        self.head = nn.Linear(  # Predict pose deltas: (Δx, Δy, Δz, Δq)
            self.hidden_size, self.output_dim
        )

    def forward(self, x):
        out, _ = self.rnn(x)  # (batch, seq_len, hidden_dim or 2*hidden_dim)
        out = self.norm(out)
        out = self.dropout(out)
        return self.head(out)
