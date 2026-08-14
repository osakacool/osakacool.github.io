"""
model/encoder.py
Section 4.1: Multi-Source Data Encoding and Shared Latent Space
Implements Equation 5: z(t) = E_θ(x(t)) = ReLU(W2·ReLU(W1·x(t)+b1)+b2)
"""

import torch
import torch.nn as nn
from config import CFG


class MultiSourceEncoder(nn.Module):
    """
    Fully connected encoder network E_θ: R^p → R^d
    
    Args from Section 5.2:
        - input_dim p = n_acc + n_strain + n_disp (multi-source channels)
        - hidden_dim h = 128
        - latent_dim d = 64
        - Activation: ReLU
    """
    def __init__(self, input_dim=None, hidden_dim=None, latent_dim=None):
        super().__init__()
        self.input_dim = input_dim or 5  # acc, strain, disp, temp, humidity
        self.hidden_dim = hidden_dim or CFG.ENCODER_HIDDEN_DIM   # 128
        self.latent_dim = latent_dim or CFG.LATENT_DIM           # 64
        
        # Eq. 5: Two fully connected layers with ReLU
        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.latent_dim)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        """
        Args:
            x: (batch, input_dim) raw sensor readings at time t
        Returns:
            z: (batch, latent_dim) shared latent representation
        """
        # Eq. 5: z(t) = ReLU(W2 · ReLU(W1·x(t) + b1) + b2)
        h = self.relu(self.fc1(x))
        z = self.relu(self.fc2(h))
        return z