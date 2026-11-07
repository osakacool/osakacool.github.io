import torch.nn.functional as F
from torch import nn

class CausalBlock(nn.Module):
    def __init__(self, inp, out, kernel, dilation, dropout):
        super().__init__()
        self.left = (kernel-1)*dilation
        self.conv = nn.Conv1d(inp, out, kernel, dilation=dilation)
        self.skip = nn.Conv1d(inp, out, 1) if inp != out else nn.Identity()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(F.relu(self.conv(F.pad(x, (self.left, 0))) + self.skip(x)))

class TCN(nn.Module):
    def __init__(self, input_dim=128, hidden=128, dilations=(1,2,4,8,16,32,64,128),
                 kernel=3, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(*[CausalBlock(input_dim if i==0 else hidden,
                    hidden, kernel, dilation, dropout) for i, dilation in enumerate(dilations)])
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.net(x.transpose(1, 2))
        return self.head(h[:, :, -1]).squeeze(-1)
