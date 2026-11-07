from torch import nn

class AttentionLSTM(nn.Module):
    def __init__(self, input_dim=128, hidden=128, layers=2, dropout=0.2, history=32):
        super().__init__()
        self.history = history
        self.norm = nn.BatchNorm1d(input_dim)
        self.rnn = nn.LSTM(input_dim, hidden, layers, batch_first=True,
                           dropout=dropout if layers > 1 else 0)
        self.score = nn.Sequential(nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        x = self.norm(x.transpose(1, 2)).transpose(1, 2)
        h, _ = self.rnn(x)
        h = h[:, -self.history:]
        weights = self.score(h).softmax(dim=1)
        return self.head((weights*h).sum(dim=1)).squeeze(-1)
