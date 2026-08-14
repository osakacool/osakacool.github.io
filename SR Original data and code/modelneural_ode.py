"""
model/neural_ode.py
Section 4.2: Neural ODE for Smooth Drift
Implements Equations 6-7:
    dz(t)/dt = f_θ(z(t), u(t), t)   (Eq. 6)
    z(t) = z(t_reset) + ∫ f_θ dz    (Eq. 7)
Solver: Dormand-Prince (adaptive step), rtol=1e-3, atol=1e-5
"""

import torch
import torch.nn as nn
from config import CFG

try:
    from torchdiffeq import odeint_adjoint as odeint
    HAS_TORCHDIFFEQ = True
except ImportError:
    HAS_TORCHDIFFEQ = False
    print("WARNING: torchdiffeq not installed. Using Euler fallback.")


class ODEFunction(nn.Module):
    """
    Neural network f_θ parameterizing the ODE derivative.
    Section 5.2: 3-layer FC network with 128 hidden units per layer.
    """
    def __init__(self, state_dim=None, latent_dim=None, hidden_units=None):
        super().__init__()
        self.state_dim = state_dim or CFG.ODE_STATE_DIM       # 64
        self.latent_dim = latent_dim or CFG.LATENT_DIM        # 64
        hidden = hidden_units or CFG.ODE_HIDDEN_UNITS          # 128
        
        # Input: [z(t), u(t), t] → state_dim + latent_dim + 1
        input_dim = self.state_dim + self.latent_dim + 1
        
        # 3-layer FC network (Section 5.2)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, self.state_dim)
        )
    
    def forward(self, t, z):
        """
        Compute dz/dt = f_θ(z(t), u(t), t)
        
        Note: In practice, u(t) is injected via set_input()
        because torchdiffeq only passes (t, z).
        """
        if not hasattr(self, '_current_input'):
            self._current_input = torch.zeros(z.shape[0], self.latent_dim, device=z.device)
        
        t_expanded = torch.ones(z.shape[0], 1, device=z.device) * t
        inp = torch.cat([z, self._current_input, t_expanded], dim=-1)
        return self.net(inp)
    
    def set_input(self, u):
        """Store current encoder output for use in ODE evaluation."""
        self._current_input = u


class NeuralODEModule(nn.Module):
    """
    Complete Neural ODE module with adaptive Dormand-Prince solver.
    Models slow environmental drift (temperature, humidity effects).
    """
    def __init__(self, state_dim=None, latent_dim=None):
        super().__init__()
        self.state_dim = state_dim or CFG.ODE_STATE_DIM
        self.latent_dim = latent_dim or CFG.LATENT_DIM
        
        self.func = ODEFunction(self.state_dim, self.latent_dim)
        
        # Solver parameters from Section 5.2
        self.method = CFG.ODE_SOLVER_METHOD    # "dopri5"
        self.rtol = CFG.ODE_REL_TOL            # 1e-3
        self.atol = CFG.ODE_ABS_TOL            # 1e-5
    
    def forward(self, z0, u_sequence, t_sequence, reset_mask=None):
        """
        Integrate ODE over time sequence.
        
        Args:
            z0: (batch, state_dim) initial state (zero after reset)
            u_sequence: (batch, T, latent_dim) encoder outputs
            t_sequence: (T,) time points
            reset_mask: (batch, T) boolean, True where reset occurs
            
        Returns:
            z_traj: (batch, T, state_dim) ODE state trajectory
        """
        batch_size, T, _ = u_sequence.shape
        
        if not HAS_TORCHDIFFEQ:
            return self._euler_fallback(z0, u_sequence, t_sequence)
        
        z_current = z0
        z_traj = []
        
        for step in range(T):
            # Event-triggered reset (Section 4.5, Eq. 22)
            if reset_mask is not None and step > 0:
                mask = reset_mask[:, step].unsqueeze(-1).float()
                z_current = z_current * (1 - mask)  # Reset to zero
            
            # Set current input for ODE function
            self.func.set_input(u_sequence[:, step, :])
            
            # Integrate one step using Dormand-Prince
            t_span = t_sequence[step:step+2] if step < T-1 else \
                     torch.stack([t_sequence[step], t_sequence[step] + 0.01])
            
            z_out = odeint(
                self.func, z_current, t_span,
                method=self.method,
                rtol=self.rtol,
                atol=self.atol
            )
            z_current = z_out[-1]
            z_traj.append(z_current)
        
        return torch.stack(z_traj, dim=1)  # (batch, T, state_dim)
    
    def _euler_fallback(self, z0, u_sequence, t_sequence):
        """Simple Euler integration when torchdiffeq is unavailable."""
        batch_size, T, _ = u_sequence.shape
        z_current = z0
        z_traj = []
        dt = 0.01
        
        for step in range(T):
            self.func.set_input(u_sequence[:, step, :])
            dz = self.func(t_sequence[step], z_current)
            z_current = z_current + dz * dt
            z_traj.append(z_current)
        
        return torch.stack(z_traj, dim=1)