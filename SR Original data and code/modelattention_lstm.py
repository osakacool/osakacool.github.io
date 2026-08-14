"""
model/attention_lstm.py
Section 4.4: Optimized LSTM with Attention-Based Gating
Implements Equations 13-20:
    Standard LSTM gates (Eq. 13-18)
    Attention scores: α_i(t) = exp(q_k^T tanh(W_a z_i(t) + U_a e_i)) / Σ ...  (Eq. 19)
    Weighted input: v̂(t) = Σ α_i(t) z_i(t)                                      (Eq. 20)

Section 5.2: LSTM hidden dim = 128, attention dim = 64
"""

import torch
import torch.nn as nn
from config import CFG


class AttentionLSTM(nn.Module):
    """
    Optimized LSTM with sensor-specific attention mechanism.
    Handles discrete long-term dependencies (cumulative damage).
    """
    def __init__(self, input_dim=None, hidden_dim=None, 
                 attn_dim=None, num_channels=5):
        super().__init__()
        self.input_dim = input_dim or CFG.LATENT_DIM          # 64
        self.hidden_dim = hidden_dim or CFG.LSTM_HIDDEN_DIM   # 128
        self.attn_dim = attn_dim or CFG.ATTENTION_DIM         # 64
        self.num_channels = num_channels
        
        # === Attention Mechanism (Eq. 19-20) ===
        # q_k ∈ R^a, W_a ∈ R^{a×d}, U_a ∈ R^{a×d}
        self.q_k = nn.Parameter(torch.randn(self.attn_dim))
        self.W_a = nn.Linear(self.input_dim, self.attn_dim, bias=False)
        self.U_a = nn.Linear(self.num_channels, self.attn_dim, bias=False)
        
        # Channel embedding (one-hot → learned)
        self.channel_embedding = nn.Embedding(num_channels, self.attn_dim)
        
        # === Standard LSTM (Eq. 13-18) ===
        # Input: fused v̂(t) + damage accumulator D(t) (Eq. 22d)
        lstm_input_dim = self.input_dim + 1  # +1 for D(t)
        self.lstm = nn.LSTMCell(lstm_input_dim, self.hidden_dim)
        
        # === Vulnerability Index Head (Eq. 25) ===
        # v(t) = σ(w_v^T h_LSTM + b_v)
        self.vul_head = nn.Sequential(
            nn.Linear(self.hidden_dim, 1),
            nn.Sigmoid()
        )
        
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, v_fused, damage_accumulator=None, 
                h0=None, c0=None, reset_mask=None, beta=None):
        """
        Args:
            v_fused: (batch, T, input_dim) fused continuous representation
            damage_accumulator: (batch, T, 1) D(t) from Eq. 22c
            h0, c0: initial LSTM states
            reset_mask: (batch, T) event flags
            beta: (batch, T) retention coefficient for hierarchical reset
            
        Returns:
            vul_index: (batch, T, 1) predicted vulnerability index
            h_traj: hidden state trajectory
            attn_weights: (batch, T, num_channels) attention distribution
        """
        batch_size, T, _ = v_fused.shape
        device = v_fused.device
        
        if h0 is None:
            h = torch.zeros(batch_size, self.hidden_dim, device=device)
            c = torch.zeros(batch_size, self.hidden_dim, device=device)
        else:
            h, c = h0, c0
        
        vul_traj = []
        h_traj = []
        attn_traj = []
        
        for step in range(T):
            # === Hierarchical Reset (Section 4.5, Eq. 22b) ===
            if reset_mask is not None and step > 0:
                mask = reset_mask[:, step].unsqueeze(-1).float()
                
                # LSTM hidden state: full reset (h → 0)
                h = h * (1 - mask)
                
                # LSTM cell state: partial retention via β
                if beta is not None:
                    beta_t = beta[:, step].unsqueeze(-1)
                    c = c * (1 - mask) + c * beta_t * mask
                else:
                    c = c * (1 - mask)
            
            # === Attention Mechanism (Eq. 19-20) ===
            v_t = v_fused[:, step, :]  # (batch, input_dim)
            
            # Compute per-channel attention scores
            # Simplified: split input_dim evenly across channels
            chunk_size = self.input_dim // self.num_channels
            attn_scores = []
            
            for ch in range(self.num_channels):
                # z_i(t): channel component
                z_i = v_t[:, ch*chunk_size:(ch+1)*chunk_size]
                # Pad to match attention input dim if needed
                if z_i.shape[-1] < self.input_dim:
                    z_i = torch.nn.functional.pad(
                        z_i, (0, self.input_dim - z_i.shape[-1])
                    )
                
                # e_i: one-hot channel encoding
                e_i = torch.zeros(batch_size, self.num_channels, device=device)
                e_i[:, ch] = 1.0
                
                # Eq. 19: score = q_k^T tanh(W_a z_i + U_a e_i)
                score = torch.matmul(
                    self.q_k,
                    self.tanh(self.W_a(z_i) + self.U_a(e_i)).unsqueeze(-1)
                ).squeeze(-1)
                attn_scores.append(score)
            
            # Softmax normalization
            attn_scores = torch.stack(attn_scores, dim=-1)  # (batch, C)
            alpha = self.softmax(attn_scores)  # (batch, C)
            attn_traj.append(alpha)
            
            # Eq. 20: v̂(t) = Σ α_i(t) · z_i(t)
            # Simplified: weighted sum of chunks
            v_hat = torch.zeros_like(v_t)
            for ch in range(self.num_channels):
                start = ch * chunk_size
                end = min((ch + 1) * chunk_size, self.input_dim)
                v_hat[:, start:end] = alpha[:, ch:ch+1] * v_t[:, start:end]
            
            # === Damage Accumulator Injection (Eq. 22d) ===
            if damage_accumulator is not None:
                D_t = damage_accumulator[:, step:step+1]  # (batch, 1)
            else:
                D_t = torch.zeros(batch_size, 1, device=device)
            
            lstm_input = torch.cat([v_hat, D_t], dim=-1)  # (batch, d+1)
            
            # === LSTM Forward (Eq. 13-18) ===
            h, c = self.lstm(lstm_input, (h, c))
            
            # === Vulnerability Index (Eq. 25) ===
            vul_t = self.vul_head(h)  # (batch, 1)
            
            vul_traj.append(vul_t)
            h_traj.append(h)
        
        vul_index = torch.stack(vul_traj, dim=1)     # (batch, T, 1)
        attn_weights = torch.stack(attn_traj, dim=1)  # (batch, T, C)
        
        return vul_index, h_traj, attn_weights