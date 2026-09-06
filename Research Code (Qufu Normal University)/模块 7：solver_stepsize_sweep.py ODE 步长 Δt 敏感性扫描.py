# solver_stepsize_sweep.py
import numpy as np

STEP_SIZE_CANDIDATES = [0.05,0.10,0.20,0.50,1.00]

def sweep_step_size(run_fn):
    """
    run_fn(dt): 使用给定dt训练模型，返回dict: macro_auc, exact_match, hamming_loss, avg_grad_norm, converge_epoch
    """
    result_list = []
    for dt in STEP_SIZE_CANDIDATES:
        print(f"Evaluate ODE step size Δt = {dt}")
        res = run_fn(dt)
        res["dt"] = dt
        result_list.append(res)
    # print table
    print("| Step Size Δt(s) | Macro AUC | Exact Match | Hamming Loss | Avg Grad Norm | Converge Epoch |")
    for item in result_list:
        print(f"| {item['dt']:.2f} | {item['macro_auc']:.3f} | {item['exact_match']:.3f} | {item['hamming_loss']:.3f} | {item['avg_grad_norm']:.2f} | {item['converge_epoch']:.0f} |")
    return result_list

"""
# 使用示例伪代码
def run_fn(dt_val):
    model = LiquidMamba(..., dt=dt_val)
    metrics, grad_norm, converge_ep = train_and_validate(model)
    return {"macro_auc":metrics["macro_auc"],
            "exact_match":metrics["exact_match"],
            "hamming_loss":metrics["hamming_loss"],
            "avg_grad_norm":grad_norm,
            "converge_epoch":converge_ep}

sweep_step_size(run_fn)
"""
