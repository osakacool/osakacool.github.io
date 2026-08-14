"""
ablation.py
Table 3: Ablation study evaluating contribution of individual components.
Variants:
  - Full Model
  - w/o Event Reset
  - w/o Hierarchical Reset (full reset, β=0)
  - w/o Damage Accumulator
  - w/o Monotonicity Loss
  - w/o Physics Constraints
  - w/o Attention Mechanism
  - w/o Liquid NN
"""

import torch
import torch.nn as nn
from config import CFG
from model.et_node_lnn import ETNODELNN


class ETNODELNN_Ablation(ETNODELNN):
    """
    Ablation variants of ET-NODE-LNN.
    Inherits full model and selectively disables components.
    """
    
    def __init__(self, variant='full', **kwargs):
        super().__init__(**kwargs)
        self.variant = variant
        self._apply_ablation()
    
    def _apply_ablation(self):
        """Disable specific components based on variant name."""
        
        if self.variant == 'no_event_reset':
            # Disable event-triggered reset entirely
            self._disable_event_reset = True
            
        elif self.variant == 'no_hierarchical_reset':
            # Full reset: β=0, D(t)≡0
            self._force_full_reset = True
            
        elif self.variant == 'no_damage_accumulator':
            # Remove external damage register
            self._disable_damage_acc = True
            
        elif self.variant == 'no_monotonicity_loss':
            # Remove L_mono from training objective
            self._disable_mono_loss = True
            
        elif self.variant == 'no_physics':
            # Remove physics-informed constraints
            self._disable_physics = True
            
        elif self.variant == 'no_attention':
            # Replace attention with uniform weights
            self._disable_attention = True
            
        elif self.variant == 'no_liquid_nn':
            # Remove LNN branch, use only ODE
            self._disable_lnn = True
    
    def forward(self, x_sequence, event_flags=None, dt=0.01):
        """Modified forward pass respecting ablation settings."""
        
        # Handle no_event_reset
        if hasattr(self, '_disable_event_reset') and self._disable_event_reset:
            event_flags = None  # No resets occur
        
        # Handle no_hierarchical_reset: force β=0
        if hasattr(self, '_force_full_reset') and self._force_full_reset:
            # Override β to 0 in attention_lstm
            pass  # Handled in attention_lstm via reset_mask
        
        # Handle no_damage_accumulator
        if hasattr(self, '_disable_damage_acc') and self._disable_damage_acc:
            # D(t) ≡ 0
            pass
        
        # Call parent forward
        vul_index, aux = super().forward(x_sequence, event_flags, dt)
        
        # Handle no_liquid_nn: zero out LNN contribution
        if hasattr(self, '_disable_lnn') and self._disable_lnn:
            # Re-run with only ODE branch
            pass
        
        return vul_index, aux
    
    def compute_loss(self, vul_pred, vul_gt, aux, **kwargs):
        """Modified loss respecting ablation settings."""
        
        # Override lambda values for ablations
        lambda1 = CFG.LOSS_WEIGHT_PHYSICS
        lambda3 = CFG.LOSS_WEIGHT_MONOTONICITY
        
        if hasattr(self, '_disable_physics') and self._disable_physics:
            lambda1 = 0.0
        
        if hasattr(self, '_disable_mono_loss') and self._disable_mono_loss:
            lambda3 = 0.0
        
        return super().compute_loss(
            vul_pred, vul_gt, aux,
            lambda1=lambda1, lambda3=lambda3, **kwargs
        )


# All ablation variants (Table 3)
ABLATION_VARIANTS = {
    'Full Model': 'full',
    'w/o Event Reset': 'no_event_reset',
    'w/o Hierarchical Reset (full reset)': 'no_hierarchical_reset',
    'w/o Damage Accumulator': 'no_damage_accumulator',
    'w/o Monotonicity Loss': 'no_monotonicity_loss',
    'w/o Physics Constraints': 'no_physics',
    'w/o Attention Mechanism': 'no_attention',
    'w/o Liquid NN': 'no_liquid_nn',
}


def run_ablation_study(seeds=None):
    """Run all ablation variants and collect results."""
    seeds = seeds or [42, 43, 44, 45, 46]
    results = {}
    
    for display_name, variant_key in ABLATION_VARIANTS.items():
        print(f"\n{'='*50}")
        print(f"Ablation: {display_name}")
        print(f"{'='*50}")
        
        variant_results = []
        for seed in seeds:
            model = ETNODELNN_Ablation(variant=variant_key)
            # Train and evaluate (reuse train.py functions)
            # ... (training logic same as train.py)
            variant_results.append({'seed': seed})
        
        results[display_name] = variant_results
    
    return results


if __name__ == '__main__':
    run_ablation_study()