"""
model/gated_fusion.py
Section 4.3: Gated Fusion of Continuous-Time Representations
Implements Equations 11-12:
    v(t) = g(t) ⊙ z(t) + (1 - g(t)) ⊙ x(t)   (Eq. 11)
    g(t) = σ(W_g[z(t); x(t)] + b_g)            (Eq. 12)
"""

import torch
import torch.nn as nn
from config import CFG


class GatedFusion(nn.Module):
    """
    Learnable gating mechanism that dynamically balances
    slow drift (ODE) and rapid response (LNN).
    
    Section 5.2: Single linear layer with sigmoid activation.
    """
    def __init__(self, state_dim=None):
        super().__init__()
        self.state_dim = state_dim or CFG.LATENT_DIM  # 64
        
        # Eq. 12: W_g ∈ R^{d×2d}, b_g ∈ R^d
        self.gate_linear = nn.Linear(2 * self.state_dim, self.state_dim)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, z_ode, x_lnn):
        """
        Args:
            z_ode: (batch, T, state_dim) Neural ODE output (slow drift)
            x_lnn: (batch, T, state_dim) LNN output (rapid response)
            
        Returns:
            v: (batch, T, state_dim) fused representation
            g: (batch, T, state_dim) gate values (for interpretability)
        """
        # Eq. 12: g(t) = σ(W_g[z(t); x(t)] + b_g)
        concat = torch.cat([z_ode, x_lnn], dim=-1)  # (batch, T, 2d)
        g = self.sigmoid(self.gate_linear(concat))   # (batch, T, d)
        
        # Eq. 11: v(t) = g(t) ⊙ z(t) + (1 - g(t)) ⊙ x(t)
        v = g * z_ode + (1 - g) * x_lnn
        
        return v, g