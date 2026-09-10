"""
run_experiment.py
ONE‑CLICK full reproduction script for journal submission
执行：多种子主模型训练、基线、消融；写日志；保存csv指标
"""
import sys
import csv
import pandas as pd
from datetime import datetime

from config import *
from 01_sim_dataset import generate_simulation_dataset, spatial_block_split
from 02_model_def import MambaPINNFramework, loss_total_calc
from metrics_utils import *
from baseline_train import run_all_baselines
from 03_1_ablation_run import run_single_ablation

def write_log(msg:str):
    t = datetime.now().strftime("%Y‑%m‑%d %H:%M:%S")
    line = f"[{t}] {msg}"
    print(line)
    with open(TRAIN_LOG_PATH,"a",encoding="utf‑8") as f:
        f.write(line + "\n")

def train_main_one_seed(seed:int):
    set_seed(seed)
    write_log(f"==== Start main experiment seed={seed} ====")
    dataset = generate_simulation_dataset(n_spatial=N_SPATIAL, n_time=N_TIME, seed=seed)
    tr_idx, te_idx = spatial_block_split(dataset, test_size=TEST_SIZE, random_state=seed)
    X_all = dataset["X"]
    Y_all = dataset["Y_obs"]
    X_train_raw = X_all[:, tr_idx, :]
    Y_train_raw = Y_all[:, tr_idx]
    X_test_raw  = X_all[:, te_idx, :]
    Y_test_raw  = Y_all[:, te_idx]

    # reshape [T,S,D] → [S,T,D]
    X_tr = X_train_raw.permute(1,0,2)[:500,:,:]
    Y_tr = Y_train_raw.permute(1,0)[:500,:]
    X_te = X_test_raw.permute(1,0,2)[:150,:,:]
    Y_te = Y_test_raw.permute(1,0)[:150,:]

    feat_dim = X_tr.shape[-1]
    model = MambaPINNFramework(feat_dim=feat_dim).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    collocation_t = torch.rand(N_COLLOCATION, device=DEVICE, requires_grad=True)
    collocation_x = torch.rand(N_COLLOCATION, device=DEVICE, requires_grad=True)
    collocation_y = torch.rand(N_COLLOCATION, device=DEVICE, requires_grad=True)

    for epoch in range(EPOCHS_MAIN):
        model.train()
        optimizer.zero_grad()
        loss_t, loss_d, loss_p, l1, l2 = loss_total_calc(model, X_tr, Y_tr, collocation_t, collocation_x, collocation_y)
        loss_t.backward()
        optimizer.step()
        if epoch %20 ==0:
            write_log(f"seed{seed} epoch{epoch:3d} total_loss={float(loss_t):.5f} data_loss={float(loss_d):.5f} phys_loss={float(loss_p):.5f} λ2={float(l2):.3f}")

    # test evaluation
    model.eval()
    with torch.no_grad():
        y_pred_test = model(X_te)
    yt_np = torch_to_np(Y_te)
    yp_np = torch_to_np(y_pred_test)
    rmse_val = rmse(yt_np, yp_np)
    rrmse_val = rrmse(yt_np, yp_np)
    r2_val = r2_score_custom(yt_np, yp_np)
    pc_val = pc_index(float(loss_p.detach().cpu()))

    return {
        "seed":seed,
        "rmse":rmse_val,
        "rrmse":rrmse_val,
        "r2":r2_val,
        "pcindex":pc_val
    }


def main():
    write_log("========== Full experiment pipeline START ==========")
    # 1 run main model multi‑seed
    main_records = []
    for s in RANDOM_SEEDS:
        res = train_main_one_seed(s)
        main_records.append(res)
    df_main = pd.DataFrame(main_records)
    df_main.to_csv(MAIN_RESULT_CSV, index=False, encoding="utf‑8")
    write_log(f"Main experiment saved {MAIN_RESULT_CSV}")

    # aggregate main result with 95% CI
    mean_rmse, lci_rmse, uci_rmse = compute_95ci(df_main["rmse"].values)
    mean_r2, lci_r2, uci_r2 = compute_95ci(df_main["r2"].values)
    mean_pc, lci_pc, uci_pc = compute_95ci(df_main["pcindex"].values)
    write_log(f"Main model aggregate: RMSE={mean_rmse:.4f} [{lci_rmse:.4f},{uci_rmse:.4f}], R2={mean_r2:.4f}, PC‑Index={mean_pc:.4f}")

    # 2 run baselines
    write_log("---------- Run baseline models ----------")
    df_base_raw, df_base_agg = run_all_baselines(RANDOM_SEEDS)

    #3 run ablation
    write_log("---------- Run ablation study ----------")
    from 03_1_ablation_run import run_single_ablation
    dataset = generate_simulation_dataset(n_spatial=N_SPATIAL,n_time=N_TIME,seed=42)
    tr_idx, te_idx = spatial_block_split(dataset)
    X_all = dataset["X"]
    Y_all = dataset["Y_obs"]
    X_train_raw = X_all[:,tr_idx,:]
    Y_train_raw = Y_all[:,tr_idx]
    X_test_raw = X_all[:,te_idx,:]
    Y_test_raw = Y_all[:,te_idx]
    X_tr = X_train_raw.permute(1,0,2)[:400,:,:]
    Y_tr = Y_train_raw.permute(1,0)[:400,:]
    X_te = X_test_raw.permute(1,0,2)[:120,:,:]
    Y_te = Y_test_raw.permute(1,0)[:120,:]
    X_tr_no_gis = X_train_raw[:,:,0:-2].permute(1,0,2)[:400,:,:]
    X_te_no_gis = X_test_raw[:,:,0:-2].permute(1,0,2)[:120,:,:]

    ablation_configs = [
        {"name":"Full Framework", "type":"full", "X_tr":X_tr, "X_te":X_te},
        {"name":"Pure PINN (w/o Mamba)", "type":"pure_pinn_no_mamba", "X_tr":X_tr, "X_te":X_te},
        {"name":"Pure Mamba (w/o PINN)", "type":"pure_mamba_no_pinn", "X_tr":X_tr, "X_te":X_te},
        {"name":"w/o GIS integration", "type":"full", "X_tr":X_tr_no_gis, "X_te":X_te_no_gis},
        {"name":"w/o adaptive weighting", "type":"fixed_lambda", "X_tr":X_tr, "X_te":X_te},
    ]
    abl_out = []
    for cfg in ablation_configs:
        r = run_single_ablation(cfg["name"],cfg["type"],cfg["X_tr"],Y_tr,cfg["X_te"],Y_te,EPOCHS_ABLATION)
        abl_out.append(r)
    pd.DataFrame(abl_out).to_csv(ABLATION_CSV,index=False,encoding="utf‑8")
    write_log(f"Ablation result saved {ABLATION_CSV}")

    write_log("========== All training finished, run plotting scripts to generate PDF figures ==========")

if __name__ == "__main__":
    main()
#（注：内容由AI生成）
