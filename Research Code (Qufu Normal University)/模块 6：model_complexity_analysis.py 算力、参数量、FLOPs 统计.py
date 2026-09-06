# model_complexity_analysis.py
import torch
import time
from thop import profile
from liquid_mamba_model import LiquidMamba

def compute_complexity(model, input_shape=(8,3000,128), device="cuda"):
    """input_shape: B=8, T=3000, feat_dim=128"""
    dummy_x = torch.randn(*input_shape).to(device)
    model = model.to(device)
    model.eval()
    macs, params = profile(model, inputs=(dummy_x,), verbose=False)
    flops = macs * 2 / 1e9  # G‑FLOPs

    # inference latency
    warmup = 3
    for _ in range(warmup):
        _ = model(dummy_x)
    torch.cuda.synchronize()
    t0 = time.time()
    repeat = 10
    for _ in range(repeat):
        _ = model(dummy_x)
    torch.cuda.synchronize()
    avg_latency_ms = (time.time()-t0)/repeat * 1000 / input_shape[0]

    total_params_m = params / 1e6
    return {
        "params(M)": round(total_params_m,2),
        "FLOPs(G)": round(flops,2),
        "infer_latency_ms_per_sample": round(avg_latency_ms,2)
    }


if __name__ == "__main__":
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = LiquidMamba(feat_dim=128, ltc_hidden=256, mamba_dim=256)
    stat = compute_complexity(net, input_shape=(8,3000,128), device=dev)
    print(stat)
