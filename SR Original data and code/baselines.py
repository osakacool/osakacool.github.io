"""
baselines.py
Section 5.3: Five baseline models for comparative evaluation.
All baselines use the same input/output interface as ET-NODE-LNN.
"""

import torch
import torch.nn as nn
from config import CFG
from model.encoder import MultiSourceEncoder
from model.neural_ode import NeuralODEModule
from model.liquid_nn import LiquidNeuralNetwork
from model.physics_loss import PhysicsInformedLoss


# ============================================================
# Baseline 1: Standard LSTM
# "operates directly on raw multi-source sensor data"
# ============================================================
class StandardLSTM(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=128):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x_sequence, **kwargs):
        h, _ = self.lstm(x_sequence)
        vul = self.head(h)
        return vul, {}


# ============================================================
# Baseline 2: Bidirectional LSTM (BiLSTM)
# "processes input in both forward and backward directions"
# ============================================================
class BiLSTM(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=128):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, 
                           bidirectional=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x_sequence, **kwargs):
        h, _ = self.lstm(x_sequence)
        vul = self.head(h)
        return vul, {}


# ============================================================
# Baseline 3: Neural ODE (replaces LSTM entirely)
# "using only continuous-time formulation"
# ============================================================
class NeuralODEBaseline(nn.Module):
    def __init__(self, input_dim=5, latent_dim=64):
        super().__init__()
        self.encoder = MultiSourceEncoder(input_dim, 128, latent_dim)
        self.ode = NeuralODEModule(latent_dim, latent_dim)
        self.head = nn.Sequential(
            nn.Linear(latent_dim, 1),
            nn.Sigmoid()
        )
        self.physics_loss = PhysicsInformedLoss()
    
    def forward(self, x_sequence, dt=0.01, **kwargs):
        batch_size, T, _ = x_sequence.shape
        u_seq = torch.stack([self.encoder(x_sequence[:, t, :]) for t in range(T)], dim=1)
        z0 = torch.zeros(batch_size, CFG.ODE_STATE_DIM, device=x_sequence.device)
        t_seq = torch.arange(T, device=x_sequence.device, dtype=torch.float32) * dt
        z_traj = self.ode(z0, u_seq, t_seq)
        vul = self.head(z_traj)
        return vul, {'z_ode': z_traj}


# ============================================================
# Baseline 4: Liquid NN (standalone)
# "without neural ODE or LSTM components"
# ============================================================
class LiquidNNBaseline(nn.Module):
    def __init__(self, input_dim=5, state_dim=64):
        super().__init__()
        self.encoder = MultiSourceEncoder(input_dim, 128, state_dim)
        self.lnn = LiquidNeuralNetwork(state_dim, state_dim, rank=16)
        self.head = nn.Sequential(
            nn.Linear(state_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x_sequence, dt=0.01, **kwargs):
        batch_size, T, _ = x_sequence.shape
        u_seq = torch.stack([self.encoder(x_sequence[:, t, :]) for t in range(T)], dim=1)
        x0 = torch.zeros(batch_size, CFG.LNN_STATE_DIM, device=x_sequence.device)
        x_traj = self.lnn(u_seq, x0, dt=dt)
        vul = self.head(x_traj)
        return vul, {}


# ============================================================
# Baseline 5: Physics-Informed LSTM (PI-LSTM)
# "incorporates modal frequency constraints into LSTM loss"
# ============================================================
class PILSTM(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=128):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        self.physics_loss = PhysicsInformedLoss()
    
    def forward(self, x_sequence, **kwargs):
        h, _ = self.lstm(x_sequence)
        vul = self.head(h)
        return vul, {'lstm_hidden': h}


# ============================================================
# Factory function
# ============================================================
def get_model(model_name, input_dim=5):
    """Get model by name for fair comparison."""
    models = {
        'standard_lstm': StandardLSTM(input_dim),
        'bilstm': BiLSTM(input_dim),
        'neural_ode': NeuralODEBaseline(input_dim),
        'liquid_nn': LiquidNNBaseline(input_dim),
        'pi_lstm': PILSTM(input_dim),
    }
    return models.get(model_name.lower())