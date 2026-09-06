# ode_adjoint_wrapper.py
import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint, odeint

class LTCAdjointWrapper(nn.Module):
    def __init__(self, ltc_ode_func, use_adjoint: bool = True):
        super().__init__()
        self.ode_func = ltc_ode_func
        self.use_adjoint = use_adjoint
        self.t_span = torch.tensor([0.0, 0.1])  # dt=0.1s

    def one_step(self, h_prev, x_t):
        """单步ODE求解；use_adjoint=True 使用伴随灵敏度反向传播，节省显存"""
        B = h_prev.shape[0]
        # concat state + input
        z0 = torch.cat([h_prev, x_t], dim=-1)
        if self.use_adjoint:
            z_out = odeint_adjoint(self.ode_func, z0, self.t_span, method="rk4")
        else:
            z_out = odeint(self.ode_func, z0, self.t_span, method="rk4")
        h_next = z_out[-1, :, :h_prev.shape[-1]]
        return h_next

    def forward_sequence(self, x_seq, h0):
        """逐时间步处理完整序列 x_seq [B, T, D]"""
        B, T, D = x_seq.shape
        h = h0
        outputs = []
        for t_idx in range(T):
            xt = x_seq[:, t_idx, :]
            h = self.one_step(h, xt)
            outputs.append(h.unsqueeze(1))
        return torch.cat(outputs, dim=1)
