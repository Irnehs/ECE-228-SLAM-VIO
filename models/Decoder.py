import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class Decoder(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=128, output_dim=7, num_layers=1, dropout=0.3):
        super().__init__()
        self.rnn = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout 
        )
        self.out = nn.Linear(hidden_dim, output_dim)

    def forward(self, context_seq, prediction_len=None):
        # CHECK THIS factor in prediction_len for variable output length
        output, _ = self.rnn(context_seq)
        pred = self.out(output)          
        return pred