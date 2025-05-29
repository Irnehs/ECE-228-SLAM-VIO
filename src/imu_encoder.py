import torch.nn as nn

class IMUEncoder(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=64, num_layers=1):
        super().__init__()
        self.rnn = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)

    def forward(self, x):
        output, (hn, _) = self.rnn(x)
        return hn[-1]
