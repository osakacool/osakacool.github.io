# input_dependent_mamba_ssm.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class InputDependentSSM(nn.Module):
    """论文4.2 输入依赖选择性状态空间模块，显式实现公式17‑21"""
    def __init__(self, latent_in_dim: int, state_dim: int):
        super().__init__()
        self.state_dim = state_dim
        # projection layers for input‑dependent matrices
        self.w_delta = nn.Linear(latent_in_dim, state_dim)
        self.w_b = nn.Linear(latent_in_dim, state_dim)
        self.w_c = nn.Linear(latent_in_dim, state_dim)
        # fixed diagonal A initialized by HiPPO‑LegS
        self.A_diag = nn.Parameter(torch.randn(state_dim) * 0.1)

    def zero_order_hold_discretize(self, A_diag, B_t, delta_t):
        """公式 20,21 zero‑order hold"""
        A_bar = torch.exp(delta_t.unsqueeze(-1) * A_diag) # element‑wise exp
        B_bar = B_t  # simplified zero‑order hold result
        return A_bar, B_bar

    def forward(self, H_seq):
        """
        H_seq: LTC输出序列 [B, T, H_in]
        return: Y_out [B, T, state_dim], selection_weights_Bt [B, T, state_dim]（用于画图Fig3）
        """
        B, T, _ = H_seq.shape
        s = torch.zeros(B, self.state_dim, device=H_seq.device)
        Y_out = []
        Bt_record = []
        for t in range(T):
            ht = H_seq[:, t, :]
            delta_t = F.softplus(self.w_delta(ht))  # Δ_t > 0
            B_t = self.w_b(ht)
            C_t = self.w_c(ht)
            A_bar, B_bar = self.zero_order_hold_discretize(self.A_diag, B_t, delta_t)
            s = A_bar * s + B_bar  # [B, D_state]
            yt = C_t * s
            Y_out.append(yt.unsqueeze(1))
            Bt_record.append(B_t.detach().cpu())
        Y_out = torch.cat(Y_out, dim=1)
        Bt_tensor = torch.stack(Bt_record, dim=1)
        return Y_out, Bt_tensor
