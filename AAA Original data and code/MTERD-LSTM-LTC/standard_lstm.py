from torch import nn

class StandardLSTM(nn.Module):
    def __init__(self, input_dim=128, hidden=128, layers=2, dropout=0.2):
        super().__init__()
        self.norm = nn.BatchNorm1d(input_dim)
        self.rnn = nn.LSTM(input_dim, hidden, layers, batch_first=True,
                           dropout=dropout if layers > 1 else 0)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        x = self.norm(x.transpose(1, 2)).transpose(1, 2)
        h, _ = self.rnn(x)
        return self.head(h[:, -1]).squeeze(-1)
