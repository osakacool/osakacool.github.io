import torch
import torch.nn as nn

class LSTM_LTC_Cell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        
        # 标准 LSTM 权重矩阵
        self.Wf = nn.Parameter(torch.Tensor(input_size + hidden_size, hidden_size))
        self.Wi = nn.Parameter(torch.Tensor(input_size + hidden_size, hidden_size))
        self.Wc = nn.Parameter(torch.Tensor(input_size + hidden_size, hidden_size))
        self.Wo = nn.Parameter(torch.Tensor(input_size + hidden_size, hidden_size))
        
        # Eq. 8: 时间常数 tau 的自适应权重
        # 3 个门 (forget, input, output) 各自拥有独立的 tau
        self.W_tau = nn.Parameter(torch.Tensor(input_size + hidden_size, hidden_size * 3))
        self.b_tau = nn.Parameter(torch.Tensor(hidden_size * 3))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W_tau)
        nn.init.zeros_(self.b_tau)
        # 其他权重初始化省略...

    def forward(self, x, h_prev, c_prev):
        combined = torch.cat((x, h_prev), dim=1)
        
        # Eq. 8: 计算神经元级别的自适应时间常数 tau (范围 0.1 ~ 10)
        tau_logits = torch.matmul(combined, self.W_tau) + self.b_tau
        tau = 0.1 + 9.9 * torch.sigmoid(tau_logits) 
        tau_f, tau_i, tau_o = torch.split(tau, self.hidden_size, dim=1)
        
        # 基础门控计算
        f_t = torch.sigmoid(torch.matmul(combined, self.Wf))
        i_t = torch.sigmoid(torch.matmul(combined, self.Wi))
        c_t = torch.tanh(torch.matmul(combined, self.Wc))
        o_t = torch.sigmoid(torch.matmul(combined, self.Wo))
        
        # Eq. 9: LTC 机制引入指数衰减调制 (假设 dt = 1 帧)
        decay_f = torch.exp(-1.0 / tau_f)
        decay_i = torch.exp(-1.0 / tau_i)
        decay_o = torch.exp(-1.0 / tau_o)
        
        f_t_eff = f_t * decay_f
        i_t_eff = i_t * decay_i
        o_t_eff = o_t * decay_o
        
        # 状态更新
        c_next = f_t_eff * c_prev + i_t_eff * c_t
        h_next = o_t_eff * torch.tanh(c_next)
        
        return h_next, c_next, tau_f, tau_i, tau_o

class LSTM_LTC_Model(nn.Module):
    def __init__(self, input_size=128, hidden_size=128, num_layers=2):
        super().__init__()
        # 论文 5.4 节：2层 LSTM-LTC，每层 128 hidden units (总计 256 neurons)
        self.cells = nn.ModuleList([
            LSTM_LTC_Cell(input_size if i == 0 else hidden_size, hidden_size)
            for i in range(num_layers)
        ])
        self.bn = nn.BatchNorm1d(input_size)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_size, 1) # Eq. 12: 线性映射到标量

    def forward(self, x):
        # x: (batch, seq_len=300, input_size=128)
        x = self.bn(x.transpose(1, 2)).transpose(1, 2)
        batch_size, seq_len, _ = x.size()
        
        h = [torch.zeros(batch_size, cell.hidden_size, device=x.device) for cell in self.cells]
        c = [torch.zeros(batch_size, cell.hidden_size, device=x.device) for cell in self.cells]
        
        all_taus = [] # 用于后续分析 Eq. 11 的有效记忆范围
        for t in range(seq_len):
            xt = x[:, t, :]
            for l, cell in enumerate(self.cells):
                inp = xt if l == 0 else h[l-1]
                h[l], c[l], tau_f, tau_i, tau_o = cell(inp, h[l], c[l])
                if l == len(self.cells) - 1:
                    all_taus.append(tau_f) # 记录最后一层的 tau
            xt = self.dropout(h[-1])
            
        out = self.fc(h[-1])
        # 返回预测值和所有时间步的 tau (用于计算 Memory Horizon)
        return out, torch.stack(all_taus, dim=1) 