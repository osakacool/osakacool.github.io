"""
data_generator.py
Replaces the Finite Element simulation process described in Section 5.1.
Generates synthetic training/testing data consistent with the manuscript.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
import scipy.signal as signal

class SyntheticTimberHallDataset(Dataset):
    def __init__(self, duration_hours=24, sampling_rate=100, num_events=3, seed=42):
        super().__init__()
        self.sr = sampling_rate
        self.num_steps = int(duration_hours * 3600 * sampling_rate)
        self.rng = np.random.default_rng(seed)
        
        # Generate base signals
        self.time = np.arange(self.num_steps) / self.sr
        self._generate_environment()
        self._generate_structural_response(num_events)
        self._compute_ground_truth()
        
    def _generate_environment(self):
        """Simulates temperature/humidity drift affecting stiffness."""
        daily_cycle = np.sin(2 * np.pi * self.time / 86400)
        self.temperature = 15 + 5 * daily_cycle + self.rng.normal(0, 0.5, self.num_steps)
        self.humidity = 60 + 10 * np.cos(2 * np.pi * self.time / 86400) + self.rng.normal(0, 2, self.num_steps)
        
    def _generate_structural_response(self, num_events):
        """Simulates ambient vibration + seismic events."""
        # Ambient: band-limited noise (0.5-10Hz typical for timber halls)
        ambient = self.rng.normal(0, 0.005, self.num_steps)
        b, a = signal.butter(4, [0.5/(self.sr/2), 10/(self.sr/2)], btype='band')
        self.acc_roof = signal.filtfilt(b, a, ambient)
        self.strain_joint = signal.filtfilt(b, a, ambient) * 10
        self.disp_base = signal.filtfilt(b, a, ambient) * 0.5
        
        self.event_mask = np.zeros(self.num_steps, dtype=bool)
        self.pga_values = []
        
        # Inject seismic events
        event_starts = sorted(self.rng.choice(
            range(int(0.1*self.num_steps), int(0.9*self.num_steps)), 
            size=num_events, replace=False
        ))
        
        for start in event_starts:
            pga = self.rng.uniform(0.05, 0.6)
            duration = int(self.rng.uniform(30, 90) * self.sr)
            end = min(start + duration, self.num_steps)
            
            # Nonlinear seismic response simulation
            t_rel = np.arange(end - start) / self.sr
            decay = np.exp(-2.5 * t_rel)
            nonlinear_resp = pga * decay * np.sin(2*np.pi*2.5*t_rel) * \
                           (1 + 0.3*np.sin(2*np.pi*8*t_rel))
            
            self.acc_roof[start:end] += nonlinear_resp
            self.strain_joint[start:end] += nonlinear_resp * 50
            self.disp_base[start:end] += np.abs(nonlinear_resp) * 2
            self.event_mask[start:end] = True
            self.pga_values.append(pga)
            
    def _compute_ground_truth(self):
        """Computes vulnerability index based on Park-Ang inspired damage model."""
        self.vul_index = np.full(self.num_steps, 0.05)
        cumulative_damage = 0.05
        
        # Find event boundaries
        changes = np.diff(self.event_mask.astype(int))
        starts = np.where(changes == 1)[0] + 1
        ends = np.where(changes == -1)[0] + 1
        
        for s, e in zip(starts, ends):
            pga = np.max(np.abs(self.acc_roof[s:e]))
            energy = np.sum(self.acc_roof[s:e]**2)
            damage_increment = 0.3 * pga + 0.01 * energy
            
            # During event
            self.vul_index[s:e] = cumulative_damage + damage_increment * \
                                  np.exp(-np.arange(e-s)/(20*self.sr))
            # After event: permanent damage accumulation
            cumulative_damage += damage_increment * 0.4
            self.vul_index[e:] = np.maximum(self.vul_index[e:], cumulative_damage)
            
        # Normalize to [0, 1]
        self.vul_index = (self.vul_index - self.vul_index.min()) / \
                         (self.vul_index.max() - self.vul_index.min() + 1e-8)
    
    def __len__(self):
        return self.num_steps
    
    def __getitem__(self, idx):
        features = np.array([
            self.acc_roof[idx],
            self.strain_joint[idx],
            self.disp_base[idx],
            self.temperature[idx],
            self.humidity[idx]
        ], dtype=np.float32)
        
        label = self.vul_index[idx]
        is_event = float(self.event_mask[idx])
        
        return torch.tensor(features), torch.tensor(label), torch.tensor(is_event)