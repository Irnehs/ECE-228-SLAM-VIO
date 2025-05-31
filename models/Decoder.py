import torch
import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=128, output_dim=7, dropout=0.3):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # LSTMCell: input_size=input_dim, hidden_size=hidden_dim
        self.rnn_cell = nn.LSTMCell(input_dim, hidden_dim)
        self.out      = nn.Linear(hidden_dim, output_dim)
        self.dropout  = nn.Dropout(dropout)

    def forward(self, context_seq, prediction_len):
        """
        context_seq:  [B, T, input_dim]
        prediction_len:  how many future steps to generate
        """

        B = context_seq.size(0)
        # Take the last timestep of the context as initial “input”
        summary = context_seq[:, -1, :]           # [B, input_dim]

        # 1) Initialize hidden and cell to zeros of shape [B, hidden_dim]
        device = context_seq.device
        h = torch.zeros(B, self.hidden_dim,  device=device)  # hidden‐state
        c = torch.zeros(B, self.hidden_dim,  device=device)  # cell‐state

        # 2) The “input_step” to the first LSTMCell call is `summary` (size=input_dim).
        input_step = summary.clone()  # [B, input_dim]

        preds = []

        for t in range(prediction_len):
            # LSTMCell(input_step, (h, c)) expects:
            #  - input_step:    [B, input_dim]
            #  - h, c:          [B, hidden_dim]
            h, c = self.rnn_cell(input_step, (h, c))
            h = self.dropout(h)

            # Project hidden state to output_dim  → this is your predicted pose
            pred = self.out(h)   # [B, output_dim]
            preds.append(pred.unsqueeze(1))  # keep a time dimension

            # 3) For next timestep, decide what to feed into LSTMCell:
            #    Here, we just feed a zero vector of size [B, input_dim].
            #    If you instead want to feed the predicted pose as input, do:
            #       input_step = pred
            #
            input_step = torch.zeros(B, self.input_dim, device=device)

        # Concatenate along the time dimension → [B, prediction_len, output_dim]
        return torch.cat(preds, dim=1)
