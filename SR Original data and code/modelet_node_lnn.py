"""
model/et_node_lnn.py
Complete ET-NODE-LNN Architecture Assembly
Integrates all modules from Sections 4.1-4.7 into a single end-to-end model.
"""

import torch
import torch.nn as nn
from config import CFG

from .encoder import MultiSourceEncoder
from .neural_ode import NeuralODEModule
from .liquid_nn import LiquidNeuralNetwork
from .gated_fusion import GatedFusion
from .attention_lstm import AttentionLSTM
from .event_trigger import EventTriggeredReset, OCSVMdetector
from .physics_loss import PhysicsInformedLoss


class ETNODELNN(nn.Module):
    """
    Event-Triggered Neural ODE-Liquid LSTM
    
    Architecture:
        Multi-source sensors → Encoder → [Neural ODE ∥ Liquid NN]
        → Gated Fusion → Attention LSTM → Vulnerability Index
        
    With: Event-triggered reset, physics-informed constraints,
          damage accumulator, hierarchical state management
    """
    def __init__(self, input_dim=5, num_channels=5):
        super().__init__()
        self.input_dim = input_dim
        self.num_channels = num_channels
        
        # === Section 4.1: Multi-Source Encoder ===
        self.encoder = MultiSourceEncoder(
            input_dim=input_dim,
            hidden_dim=CFG.ENCODER_HIDDEN_DIM,   # 128
            latent_dim=CFG.LATENT_DIM             # 64
        )
        
        # === Section 4.2: Parallel Continuous-Time Modules ===
        self.neural_ode = NeuralODEModule(
            state_dim=CFG.ODE_STATE_DIM,   # 64
            latent_dim=CFG.LATENT_DIM      # 64
        )
        self.liquid_nn = LiquidNeuralNetwork(
            input_dim=CFG.LATENT_DIM,      # 64
            state_dim=CFG.LNN_STATE_DIM,   # 64
            rank=CFG.LNN_LOW_RANK          # 16
        )
        
        # === Section 4.3: Gated Fusion ===
        self.gated_fusion = GatedFusion(
            state_dim=CFG.LATENT_DIM  # 64
        )
        
        # === Section 4.4: Attention LSTM ===
        self.attention_lstm = AttentionLSTM(
            input_dim=CFG.LATENT_DIM,       # 64
            hidden_dim=CFG.LSTM_HIDDEN_DIM, # 128
            attn_dim=CFG.ATTENTION_DIM,     # 64
            num_channels=num_channels
        )
        
        # === Section 4.5: Event-Triggered Reset ===
        self.event_reset = EventTriggeredReset(
            lstm_hidden_dim=CFG.LSTM_HIDDEN_DIM,
            smoothing_alpha=CFG.DAMAGE_ACCUMULATOR_ALPHA
        )
        
        # === Section 4.6: Physics-Informed Loss ===
        self.physics_loss = PhysicsInformedLoss(
            true_freqs=CFG.TRUE_MODAL_FREQS_HZ,
            window_length=CFG.STFT_WINDOW_LENGTH,
            sampling_rate=CFG.SAMPLING_RATE_HZ
        )
    
    def forward(self, x_sequence, event_flags=None, dt=0.01):
        """
        Full forward pass.
        
        Args:
            x_sequence: (batch, T, input_dim) multi-source sensor data
            event_flags: (batch, T) boolean, True during seismic events
            dt: time step
            
        Returns:
            vul_index: (batch, T, 1) predicted vulnerability index
            aux: dict with auxiliary outputs for loss computation
        """
        batch_size, T, _ = x_sequence.shape
        device = x_sequence.device
        
        if event_flags is None:
            event_flags = torch.zeros(batch_size, T, dtype=torch.bool, device=device)
        
        # === Step 1: Encode multi-source data (Eq. 5) ===
        u_sequence = torch.zeros(batch_size, T, CFG.LATENT_DIM, device=device)
        for t in range(T):
            u_sequence[:, t, :] = self.encoder(x_sequence[:, t, :])
        
        # === Step 2: Neural ODE - slow drift (Eq. 6-7) ===
        z0 = torch.zeros(batch_size, CFG.ODE_STATE_DIM, device=device)
        t_seq = torch.arange(T, device=device, dtype=torch.float32) * dt
        z_ode = self.neural_ode(z0, u_sequence, t_seq, reset_mask=event_flags)
        
        # === Step 3: Liquid NN - rapid response (Eq. 8-10) ===
        x0 = torch.zeros(batch_size, CFG.LNN_STATE_DIM, device=device)
        x_lnn = self.liquid_nn(u_sequence, x0, dt=dt, reset_mask=event_flags)
        
        # === Step 4: Gated Fusion (Eq. 11-12) ===
        v_fused, gate_values = self.gated_fusion(z_ode, x_lnn)
        
        # === Step 5: Damage Accumulator (Eq. 22c-22d) ===
        D = torch.zeros(batch_size, T, 1, device=device)
        D_current = torch.zeros(batch_size, device=device)
        
        for t in range(T):
            if t > 0 and event_flags[:, t].any():
                # Update D at reset instants
                vul_pre = D_current  # Simplified: use previous D
                D_current = self.event_reset.update_damage_accumulator(
                    D_current, vul_pre
                )
            D[:, t, 0] = D_current
        
        # === Step 6: Attention LSTM (Eq. 13-20) ===
        # Compute β for hierarchical reset
        vul_index, h_traj, attn_weights = self.attention_lstm(
            v_fused,
            damage_accumulator=D,
            reset_mask=event_flags
        )
        
        # === Auxiliary outputs ===
        aux = {
            'z_ode': z_ode,           # For physics loss
            'x_lnn': x_lnn,
            'gate_values': gate_values,
            'attn_weights': attn_weights,
            'damage_accumulator': D,
            'event_flags': event_flags
        }
        
        return vul_index, aux
    
    def compute_loss(self, vul_pred, vul_gt, aux, 
                     lambda1=None, lambda2=None, lambda3=None):
        """
        Combined training objective (Eq. 26):
        L = L_pred + λ₁·L_physics + λ₂·L_reg + λ₃·L_mono
        
        Section 5.2: λ₁=0.1, λ₂=1e-4, λ₃=0.05
        """
        lambda1 = lambda1 or CFG.LOSS_WEIGHT_PHYSICS       # 0.1
        lambda2 = lambda2 or CFG.LOSS_WEIGHT_REG           # 1e-4
        lambda3 = lambda3 or CFG.LOSS_WEIGHT_MONOTONICITY  # 0.05
        
        # L_pred: MSE between predicted and ground truth vulnerability
        L_pred = nn.functional.mse_loss(vul_pred, vul_gt)
        
        # L_physics: Modal frequency tracking (Eq. 24)
        L_physics = self.physics_loss(aux['z_ode'])
        
        # L_reg: L2 regularization on all parameters
        L_reg = torch.tensor(0.0, device=vul_pred.device)
        for param in self.parameters():
            L_reg = L_reg + torch.norm(param, p=2) ** 2
        
        # L_mono: Damage monotonicity (Eq. 26b)
        D = aux['damage_accumulator'].squeeze(-1)  # (batch, T)
        event_flags = aux['event_flags']
        L_mono = torch.tensor(0.0, device=vul_pred.device)
        
        for b in range(D.shape[0]):
            reset_times = torch.where(event_flags[b])[0].tolist()
            if len(reset_times) > 1:
                L_mono += self.event_reset.monotonicity_loss(D[b], reset_times)
        L_mono = L_mono / max(D.shape[0], 1)
        
        # Total loss (Eq. 26)
        total_loss = L_pred + lambda1 * L_physics + lambda2 * L_reg + lambda3 * L_mono
        
        return total_loss, {
            'L_pred': L_pred.item(),
            'L_physics': L_physics.item(),
            'L_reg': L_reg.item(),
            'L_mono': L_mono.item()
        }