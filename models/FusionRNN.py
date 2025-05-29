import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F

class FusionRNN(nn.Modulee):
    def __init__(self, input_dim=519, hidden_dim=128, bidirectional=False):
        super(FusionRNN, self).__init__()
        self.bidirectional = bidirectional
        self.hidden_size = hidden_dim * 2 if bidirectional else hidden_dim
        self.rnn = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=bidirectional)
        self.norm = nn.LayerNorm(self.hidden_size)
        self.dropout = nn.Dropout(0.3)
        self.head = nn.Linear(self.hidden_size, 7)  # Predict pose deltas: (Δx, Δy, Δz, Δq)


    def forward(self, x):
        out, _ = self.rnn(x)  # (batch, seq_len, hidden_dim or 2*hidden_dim)
        out = self.norm(out)
        out = self.dropout(out)
        return self.head(out)
