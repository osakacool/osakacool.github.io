"""Equation-based reconstruction of manuscript Equations 2-6 and 11.

This is the manuscript's exponentially modulated LSTM, not an ODE LTC cell.
Gate order is input, forget, output; all three have separate dynamic tau.
"""
import torch
from torch import nn

class LTCCell(nn.Module):
    def __init__(self, input_dim, hidden, dt=0.1, tau_min=0.1,
                 tau_max=10.0, fixed_tau=None):
        super().__init__()
        if not 0 < tau_min < tau_max or dt <= 0:
            raise ValueError('Require 0 < tau_min < tau_max and dt > 0')
        if fixed_tau is not None and fixed_tau <= 0:
            raise ValueError('fixed_tau must be positive')
        self.hidden, self.dt = hidden, dt
        self.tau_min, self.tau_max, self.fixed_tau = tau_min, tau_max, fixed_tau
        self.gates = nn.Linear(input_dim + hidden, hidden * 4)
        self.tau_map = nn.Linear(input_dim + hidden, hidden * 3)
        # At x=h=0, tau is Uniform(0.1,10), using inverse-logit biases.
        with torch.no_grad():
            initial = torch.empty(hidden * 3).uniform_(tau_min, tau_max)
            u = ((initial - tau_min) / (tau_max - tau_min)).clamp(1e-5, 1-1e-5)
            self.tau_map.bias.copy_(torch.logit(u))
            nn.init.normal_(self.tau_map.weight, std=0.01)

    def forward(self, x, state):
        h, c = state
        z = torch.cat([x, h], dim=-1)
        gi, gf, go, candidate = self.gates(z).chunk(4, dim=-1)
        tau = self.tau_min + (self.tau_max-self.tau_min) * torch.sigmoid(self.tau_map(z))
        if self.fixed_tau is not None:
            tau = torch.full_like(tau, self.fixed_tau)
        ti, tf, to = tau.chunk(3, dim=-1)
        i = torch.sigmoid(gi) * torch.exp(-self.dt/ti)
        f = torch.sigmoid(gf) * torch.exp(-self.dt/tf)
        o = torch.sigmoid(go) * torch.exp(-self.dt/to)
        c = f*c + i*torch.tanh(candidate)
        h = o*torch.tanh(c)
        return h, c, tf, f

class LSTMLTC(nn.Module):
    def __init__(self, input_dim=128, hidden=128, layers=2, dropout=0.2,
                 dt=0.1, tau_min=0.1, tau_max=10.0, fixed_tau=None):
        super().__init__()
        self.hidden, self.layers, self.dt = hidden, layers, dt
        self.norms = nn.ModuleList([nn.BatchNorm1d(input_dim if i == 0 else hidden)
                                    for i in range(layers)])
        self.cells = nn.ModuleList([LTCCell(input_dim if i == 0 else hidden,
                               hidden, dt, tau_min, tau_max, fixed_tau)
                               for i in range(layers)])
        self.dropout, self.head = nn.Dropout(dropout), nn.Linear(hidden, 1)

    def forward(self, x, return_diagnostics=False):
        # Batch normalization pools B and T only; never mixes train/test partitions.
        tau_means, retention_means = [], []
        for layer, (norm, cell) in enumerate(zip(self.norms, self.cells)):
            x = norm(x.transpose(1, 2)).transpose(1, 2)
            h = x.new_zeros(x.shape[0], self.hidden)
            c = torch.zeros_like(h)
            states, tau_sum, f_sum = [], torch.zeros_like(h), torch.zeros_like(h)
            for t in range(x.shape[1]):
                h, c, tau, f = cell(x[:, t], (h, c))
                states.append(h)
                if return_diagnostics:
                    tau_sum = tau_sum + tau
                    f_sum = f_sum + f
            x = torch.stack(states, dim=1)
            if return_diagnostics:
                tau_means.append(tau_sum / x.shape[1])
                retention_means.append(f_sum / x.shape[1])
            if layer < self.layers-1:
                x = self.dropout(x)
        prediction = self.head(x[:, -1]).squeeze(-1)
        if not return_diagnostics:
            return prediction
        # One value per neuron per layer, using forget-gate dynamics.
        tau = torch.stack(tau_means, dim=1)
        f = torch.stack(retention_means, dim=1)
        return prediction, {'forget_tau': tau, 'forget_retention': f}
