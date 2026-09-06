import torch
import torch.nn as nn
from torchdiffeq import odeint
from mamba_ssm import Mamba

class LTC_ODE(nn.Module):
    """Liquid Time‑Constant ODE module"""
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.tau_base = nn.Parameter(torch.ones(hidden_dim))
        self.f_net = nn.Sequential(
            nn.Linear(input_dim+hidden_dim,128),
            nn.Tanh(),
            nn.Linear(128, hidden_dim),
        )
        self.g_net = nn.Sequential(
            nn.Linear(input_dim+hidden_dim,128),
            nn.Tanh(),
            nn.Linear(128, hidden_dim),
        )

    def forward(self, t, state_and_input):
        h, x = torch.split(state_and_input, [self.hidden_dim, x.shape[-1]], dim=-1)
        cat = torch.cat([h, x], dim=-1)
        f = self.f_net(cat)
        g = self.g_net(cat)
        one_over_tau = 1.0 / torch.clamp(self.tau_base, min=1e-6)
        dhdt = -(one_over_tau + f) * h + g
        return torch.cat([dhdt, torch.zeros_like(x)], dim=-1)

class LTCFrontEnd(nn.Module):
    def __init__(self, feat_dim, hidden_dim=256, dt=0.1):
        super().__init__()
        self.dt = dt
        self.hidden_dim = hidden_dim
        self.ode_func = LTC_ODE(feat_dim, hidden_dim)

    def rk4_step(self, h_prev, x_t):
        concat_inp = torch.cat([h_prev, x_t], dim=-1)
        k1 = self.ode_func(0.0, concat_inp)[..., :self.hidden_dim]
        k2 = self.ode_func(0.0, torch.cat([h_prev + self.dt/2*k1, x_t], dim=-1))[..., :self.hidden_dim]
        k3 = self.ode_func(0.0, torch.cat([h_prev + self.dt/2*k2, x_t], dim=-1))[..., :self.hidden_dim]
        k4 = self.ode_func(0.0, torch.cat([h_prev + self.dt*k3, x_t], dim=-1))[..., :self.hidden_dim]
        h_next = h_prev + self.dt/6.0*(k1 + 2*k2 +2*k3 +k4)
        return h_next

    def forward(self, x_seq):
        """x_seq: [B, T, feat_dim]"""
        B,T,_ = x_seq.shape
        h = torch.zeros(B, self.hidden_dim, device=x_seq.device)
        out_list = []
        for t in range(T):
            x_t = x_seq[:,t,:]
            h = self.rk4_step(h, x_t)
            out_list.append(h.unsqueeze(1))
        H = torch.cat(out_list, dim=1) # [B,T,H]
        return H

class LiquidMamba(nn.Module):
    def __init__(self, feat_dim=128, ltc_hidden=256, mamba_dim=256, mamba_expand=2, num_sel=4, dropout=0.2):
        super().__init__()
        self.ltc = LTCFrontEnd(feat_dim, hidden_dim=ltc_hidden, dt=0.1)
        self.drop1 = nn.Dropout(dropout)
        self.mamba_blocks = nn.Sequential(*[
            Mamba(d_model=ltc_hidden, d_state=mamba_dim, expand=mamba_expand, d_conv=4)
            for _ in range(num_sel)
        ])
        self.drop2 = nn.Dropout(dropout)
        # 3 SEL classification heads
        self.head_emp = nn.Linear(ltc_hidden, 1)
        self.head_out = nn.Linear(ltc_hidden,1)
        self.head_prob = nn.Linear(ltc_hidden,1)

    def forward(self, x):
        """x: [B,T,128] log‑mel feature"""
        H = self.ltc(x)
        H = self.drop1(H)
        Y = self.mamba_blocks(H)
        Y = self.drop2(Y)
        z = torch.mean(Y, dim=1) # mean pooling over time
        y_emp = torch.sigmoid(self.head_emp(z))
        y_out = torch.sigmoid(self.head_out(z))
        y_prob = torch.sigmoid(self.head_prob(z))
        return torch.cat([y_emp,y_out,y_prob], dim=-1)
