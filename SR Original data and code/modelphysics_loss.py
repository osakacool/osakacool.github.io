"""
model/physics_loss.py
Section 4.6: Physics-Informed Constraints on the Neural ODE
Implements Equations 23-24:
    STFT: Z(t,f) = Σ z(τ)w(τ-t)e^{-j2πfτ}     (Eq. 23)
    L_physics = Σ |f_i(t) - f_i^true|²           (Eq. 24)
    
Section 5.2: First 3 natural frequencies, Hamming window, W=1024
"""

import torch
import torch.nn as nn
import numpy as np
from config import CFG


class PhysicsInformedLoss(nn.Module):
    """
    Physics-informed loss constraining ODE latent state
    to track known modal frequencies.
    """
    def __init__(self, true_freqs=None, window_length=None, 
                 sampling_rate=None, num_modes=None):
        super().__init__()
        self.true_freqs = true_freqs or CFG.TRUE_MODAL_FREQS_HZ  # [2.5, 7.8, 12.1]
        self.window_length = window_length or CFG.STFT_WINDOW_LENGTH  # 1024
        self.sampling_rate = sampling_rate or CFG.SAMPLING_RATE_HZ    # 100
        self.num_modes = num_modes or CFG.NUM_MODAL_FREQUENCIES       # 3
        
        # Hamming window (Eq. 23)
        self.register_buffer(
            'hamming_window',
            torch.hamming_window(self.window_length)
        )
    
    def extract_instantaneous_frequencies(self, z_latent):
        """
        Extract top-k instantaneous frequencies from ODE latent state
        using STFT (Eq. 23).
        
        Args:
            z_latent: (batch, T, state_dim) ODE latent trajectory
            
        Returns:
            inst_freqs: (batch, num_modes) extracted frequencies
        """
        batch_size, T, state_dim = z_latent.shape
        
        # Use first principal component of latent state for STFT
        # (simplified: use mean across dimensions)
        signal = z_latent.mean(dim=-1)  # (batch, T)
        
        # Compute STFT
        if T < self.window_length:
            # Pad if signal too short
            pad_len = self.window_length - T
            signal = torch.nn.functional.pad(signal, (0, pad_len))
            T = self.window_length
        
        # Sliding window STFT
        n_frames = T - self.window_length + 1
        if n_frames <= 0:
            n_frames = 1
        
        # Use middle frame for frequency extraction
        mid_frame = T // 2
        start = max(0, mid_frame - self.window_length // 2)
        end = min(T, start + self.window_length)
        
        windowed = signal[:, start:end] * self.hamming_window[:end-start].to(signal.device)
        
        # FFT
        spectrum = torch.fft.rfft(windowed, dim=-1)
        magnitude = torch.abs(spectrum)  # (batch, freq_bins)
        
        # Find top-3 peaks
        freq_bins = torch.fft.rfftfreq(
            self.window_length, d=1.0/self.sampling_rate
        ).to(magnitude.device)
        
        # Get indices of top-k peaks
        _, top_indices = torch.topk(magnitude, self.num_modes, dim=-1)
        
        # Convert to frequencies
        inst_freqs = freq_bins[top_indices]  # (batch, num_modes)
        
        return inst_freqs
    
    def forward(self, z_latent):
        """
        Compute physics-informed loss (Eq. 24).
        
        L_physics = Σ_{i=1}^{3} |f_i(t) - f_i^true|²
        
        Args:
            z_latent: (batch, T, state_dim) ODE latent trajectory
            
        Returns:
            loss: scalar physics-informed loss
        """
        inst_freqs = self.extract_instantaneous_frequencies(z_latent)
        
        # True frequencies as tensor
        true_freqs = torch.tensor(
            self.true_freqs, dtype=torch.float32, device=z_latent.device
        )
        
        # Eq. 24: L_physics = Σ |f_i - f_i^true|²
        # Sort both to match modes correctly
        inst_sorted, _ = torch.sort(inst_freqs, dim=-1)
        true_sorted, _ = torch.sort(true_freqs.unsqueeze(0).expand_as(inst_sorted), dim=-1)
        
        loss = torch.mean((inst_sorted - true_sorted) ** 2)
        
        return loss