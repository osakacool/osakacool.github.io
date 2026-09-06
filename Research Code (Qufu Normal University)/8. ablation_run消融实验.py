"""
消融实验：
1. w/o LTC (pure Mamba)
2. w/o Mamba(LTC+mean pool)
3. LTC + S4
4. LTC fixed time constant
5. Euler solver instead RK4
"""
import torch
from liquid_mamba_model import LiquidMamba

def run_ablation(config_list):
    for cfg in config_list:
        print(f"Running ablation: {cfg['name']}")
        # modify model hyper‑param for ablation variant
        # train and save metrics
        pass

if __name__=="__main__":
    ablation_configs = [
        {"name":"full_liquid_mamba"},
        {"name":"without_ltc"},
        {"name":"without_mamba"},
        {"name":"ltc_s4"},
        {"name":"ltc_fixed_tau"},
        {"name":"euler_solver"}
    ]
    run_ablation(ablation_configs)
