"""
model/liquid_nn.py
Section 4.2: Liquid Neural Network for Rapid Nonlinear Response
Implements Equations 8-10:
    τ dx(t)/dt = -x(t) + tanh(W_xh(t)u(t) + W_hh(t)x(t) + b_h)  (Eq. 8)
    W_xh(t) = W_xh_base + A_xh · diag(σ(B_xh[u;x] + c_xh))     (Eq. 9)
    W_hh(t) = W_hh_base + A_hh · diag(σ(B_hh[u;x] + c_hh))     (Eq. 10)
    
Section 5.2: latent dim d=64, low-rank r=16, τ_init=0.1s
"""

import torch
import torch.nn as nn
from config import CFG


class LiquidNeuralNetwork(nn.Module):
    """
    Liquid Neural Network with time-varying synaptic weights.
    Captures rapid nonlinear responses during seismic events.
    """
    def __init__(self, input_dim=None, state_dim=None, rank=None):
        super().__init__()
        self.input_dim = input_dim or CFG.LATENT_DIM      # 64
        self.state_dim = state_dim or CFG.LNN_STATE_DIM   # 64
        self.rank = rank or CFG.LNN_LOW_RANK              # 16
        
        # Learnable time constant τ (Section 5.2: initialized to 0.1)
        self.log_tau = nn.Parameter(
            torch.tensor(CFG.LNN_TIME_CONSTANT_INIT).log()
        )
        
        # === Eq. 9: Time-varying input weight W_xh(t) ===
        # W_xh_base ∈ R^{n×p}
        self.W_xh_base = nn.Parameter(
            torch.randn(self.state_dim, self.input_dim) * 0.01
        )
        # A_xh ∈ R^{n×r} (low-rank factorization)
        self.A_xh = nn.Parameter(
            torch.randn(self.state_dim, self.rank) * 0.01
        )
        # B_xh ∈ R^{r×(p+n)}, c_xh ∈ R^r
        self.B_xh = nn.Linear(self.input_dim + self.state_dim, self.rank)
        
        # === Eq. 10: Time-varying hidden weight W_hh(t) ===
        # W_hh_base ∈ R^{n×n}
        self.W_hh_base = nn.Parameter(
            torch.randn(self.state_dim, self.state_dim) * 0.01
        )
        # A_hh ∈ R^{n×r}
        self.A_hh = nn.Parameter(
            torch.randn(self.state_dim, self.rank) * 0.01
        )
        # B_hh ∈ R^{r×(p+n)}, c_hh ∈ R^r
        self.B_hh = nn.Linear(self.input_dim + self.state_dim, self.rank)
        
        # Bias b_h ∈ R^n (Eq. 8)
        self.b_h = nn.Parameter(torch.zeros(self.state_dim))
        
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, u_sequence, x0=None, dt=0.01, reset_mask=None):
        """
        Simulate LNN dynamics over a time sequence.
        
        Args:
            u_sequence: (batch, T, input_dim) encoder outputs
            x0: (batch, state_dim) initial hidden state
            dt: time step for numerical integration
            reset_mask: (batch, T) event reset flags
            
        Returns:
            x_traj: (batch, T, state_dim) LNN state trajectory
        """
        batch_size, T, _ = u_sequence.shape
        device = u_sequence.device
        
        if x0 is None:
            x = torch.zeros(batch_size, self.state_dim, device=device)
        else:
            x = x0
        
        tau = self.log_tau.exp()  # Ensure τ > 0
        x_traj = []
        
        for step in range(T):
            # Event-triggered reset (Section 4.5)
            if reset_mask is not None and step > 0:
                mask = reset_mask[:, step].unsqueeze(-1).float()
                x = x * (1 - mask)
            
            u_t = u_sequence[:, step, :]  # (batch, input_dim)
            
            # Compute time-varying weights (Eq. 9 & 10)
            concat_ux = torch.cat([u_t, x], dim=-1)  # (batch, p+n)
            
            # Eq. 9: W_xh(t) = W_xh_base + A_xh · diag(σ(B_xh[u;x] + c_xh))
            gate_xh = self.sigmoid(self.B_xh(concat_ux))  # (batch, r)
            # A_xh @ diag(gate_xh) → (batch, n, p)
            W_xh_t = self.W_xh_base.unsqueeze(0) + \
                     torch.einsum('nr,br->bnp', self.A_xh, gate_xh)
            
            # Eq. 10: W_hh(t) = W_hh_base + A_hh · diag(σ(B_hh[u;x] + c_hh))
            gate_hh = self.sigmoid(self.B_hh(concat_ux))  # (batch, r)
            W_hh_t = self.W_hh_base.unsqueeze(0) + \
                     torch.einsum('nr,br->bnp', self.A_hh, gate_hh)
            
            # Eq. 8: τ dx/dt = -x + tanh(W_xh(t)u + W_hh(t)x + b_h)
            Wx_u = torch.bmm(W_xh_t, u_t.unsqueeze(-1)).squeeze(-1)
            Wh_x = torch.bmm(W_hh_t, x.unsqueeze(-1)).squeeze(-1)
            
            dx_dt = (-x + self.tanh(Wx_u + Wh_x + self.b_h)) / tau
            
            # Euler integration step
            x = x + dx_dt * dt
            x_traj.append(x)
        
        return torch.stack(x_traj, dim=1)  # (batch, T, state_dim)