"""
model/event_trigger.py
Section 4.5: Event-Triggered Reset Mechanism
- OC-SVM anomaly detector (Eq. 21)
- Hierarchical reset (Eq. 22)
- Damage accumulator with EMA (Eq. 22c-22d)
- Hysteresis: 10 consecutive samples below threshold
"""

import torch
import torch.nn as nn
import numpy as np
from config import CFG

try:
    from sklearn.svm import OneClassSVM
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


class OCSVMdetector:
    """
    One-Class SVM anomaly detector for seismic event detection.
    
    Section 5.2:
        - RBF kernel with γ = 0.01
        - Threshold τ = 95th percentile of ambient scores
        - Hysteresis: 10 consecutive samples below τ to resume
    """
    def __init__(self, gamma=None, threshold_percentile=None, 
                 hysteresis_samples=None):
        self.gamma = gamma or CFG.OC_SVM_GAMMA                          # 0.01
        self.threshold_pct = threshold_percentile or CFG.OC_SVM_THRESHOLD_PERCENTILE  # 95
        self.hysteresis = hysteresis_samples or CFG.RESET_HYSTERESIS_SAMPLES  # 10
        self.threshold = None
        self.model = None
        
        if HAS_SKLEARN:
            self.model = OneClassSVM(
                kernel='rbf',
                gamma=self.gamma,
                nu=0.05  # Expected fraction of outliers
            )
    
    def fit(self, ambient_data):
        """
        Train OC-SVM on ambient vibration data (non-seismic).
        
        Args:
            ambient_data: (N, features) normal operating condition data
        """
        if not HAS_SKLEARN:
            raise ImportError("scikit-learn required for OC-SVM")
        
        self.model.fit(ambient_data)
        
        # Compute anomaly scores on training data
        scores = -self.model.score_samples(ambient_data)
        
        # Eq. 21 threshold: τ = 95th percentile
        self.threshold = np.percentile(scores, self.threshold_pct)
        
        return self
    
    def detect(self, acceleration_data):
        """
        Detect seismic events from acceleration signals.
        
        Args:
            acceleration_data: (T, features) acceleration time series
            
        Returns:
            event_flags: (T,) boolean array, True during seismic events
            anomaly_scores: (T,) OC-SVM anomaly scores
        """
        if not HAS_SKLEARN:
            return np.zeros(len(acceleration_data), dtype=bool), \
                   np.zeros(len(acceleration_data))
        
        # Compute anomaly scores (Eq. 21)
        scores = -self.model.score_samples(acceleration_data)
        
        # Threshold detection
        above_threshold = scores > self.threshold
        
        # Apply hysteresis (Section 4.5)
        event_flags = self._apply_hysteresis(above_threshold)
        
        return event_flags, scores
    
    def _apply_hysteresis(self, above_threshold):
        """
        Prevent rapid oscillation between reset and normal operation.
        Requires 10 consecutive samples below τ to resume normal.
        """
        flags = np.copy(above_threshold)
        in_event = False
        below_count = 0
        
        for i in range(len(flags)):
            if above_threshold[i]:
                in_event = True
                below_count = 0
            elif in_event:
                below_count += 1
                if below_count >= self.hysteresis:
                    in_event = False
                    below_count = 0
            
            flags[i] = in_event
        
        return flags


class EventTriggeredReset(nn.Module):
    """
    Hierarchical reset mechanism governing state reinitialization.
    
    Section 4.5:
        - ODE/LNN states: full reset to zero (Eq. 22)
        - LSTM cell state: partial retention via learnable β
        - Damage accumulator: EMA update (Eq. 22c)
        - β = σ(ω_β^T c_LSTM + b_β) (learnable retention)
    """
    def __init__(self, lstm_hidden_dim=None, smoothing_alpha=None):
        super().__init__()
        self.hidden_dim = lstm_hidden_dim or CFG.LSTM_HIDDEN_DIM  # 128
        self.alpha = smoothing_alpha or CFG.DAMAGE_ACCUMULATOR_ALPHA  # 0.9
        
        # Learnable β parameterization (Section 4.5)
        # β = σ(ω_β^T c_LSTM + b_β)
        self.omega_beta = nn.Parameter(torch.randn(self.hidden_dim) * 0.01)
        self.b_beta = nn.Parameter(torch.zeros(1))
        self.sigmoid = nn.Sigmoid()
    
    def compute_beta(self, c_lstm):
        """
        Compute adaptive retention coefficient β.
        
        Args:
            c_lstm: (batch, hidden_dim) LSTM cell state before reset
            
        Returns:
            beta: (batch,) retention coefficient in [0, 1]
        """
        # β = σ(ω_β^T c_LSTM + b_β)
        beta = self.sigmoid(
            torch.matmul(c_lstm, self.omega_beta) + self.b_beta
        )
        return beta
    
    def update_damage_accumulator(self, D_prev, vul_index_pre_reset):
        """
        Eq. 22c: D(t_k+) = α·D(t_k-) + (1-α)·v(t_k-)
        
        Args:
            D_prev: (batch,) previous damage accumulator value
            vul_index_pre_reset: (batch,) vulnerability index before reset
            
        Returns:
            D_new: (batch,) updated damage accumulator
        """
        D_new = self.alpha * D_prev + (1 - self.alpha) * vul_index_pre_reset
        return D_new
    
    def monotonicity_loss(self, D_sequence, reset_times):
        """
        Eq. 26b: L_mono = Σ max(0, D(t_k+) - D(t_{k+1}-))
        
        Ensures damage accumulator is non-decreasing.
        
        Args:
            D_sequence: (T,) damage accumulator trajectory
            reset_times: list of reset time indices
            
        Returns:
            loss: scalar monotonicity loss
        """
        loss = torch.tensor(0.0, device=D_sequence.device)
        
        for i in range(len(reset_times) - 1):
            t_k = reset_times[i]
            t_k1 = reset_times[i + 1]
            
            if t_k < len(D_sequence) and t_k1 < len(D_sequence):
                # Penalize decrease: max(0, D(t_k+) - D(t_{k+1}-))
                decrease = D_sequence[t_k] - D_sequence[t_k1]
                loss = loss + torch.relu(decrease)
        
        return loss