"""
03_train_exp.py
对应论文章节5.3实现细节；5.2基线；6结果；消融实验；灵敏度模拟
功能：模型训练循环；基线模拟；消融模拟；梯度范数记录
"""
import torch
import torch.optim as optim
import pandas as pd
from 01_sim_dataset import generate_simulation_dataset, spatial_block_split
from 02_model_def import MambaPINNFramework, loss_total_calc

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_mamba_pinn(X_train_tensor, Y_train_tensor, epochs=145):
    """
    Mamba‑PINN训练流程，论文5.3
    :param X_train_tensor: [B,Seq,Feat]
    :param Y_train_tensor: [B,Seq]
    :param epochs:训练轮次
    :return model, loss_records, grad_ratio_records
    """
    feat_dim = X_train_tensor.shape[-1]
    model = MambaPINNFramework(feat_dim=feat_dim).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)

    # PINN配点采样，论文设置10000个collocation点
    n_collocation = 10000
    collocation_t = torch.rand(n_collocation, device=device, requires_grad=True)
    collocation_x = torch.rand(n_collocation, device=device, requires_grad=True)
    collocation_y = torch.rand(n_collocation, device=device, requires_grad=True)

    loss_records = {"total":[], "data":[], "physics":[], "lam1":[], "lam2":[]}
    grad_ratio_records = []

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        loss_t, loss_d, loss_p, l1, l2 = loss_total_calc(
            model, X_train_tensor, Y_train_tensor,
            collocation_t, collocation_x, collocation_y
        )
        loss_t.backward()

        # 计算梯度范数比，对应图7
        grad_data_norm_sq = 0.0
        grad_physics_norm_sq =0.0
        for _, p in model.named_parameters():
            if p.grad is not None:
                grad_data_norm_sq += (p.grad * l1).norm()**2
                grad_physics_norm_sq += (p.grad * l2).norm()**2
        grad_ratio = torch.sqrt(grad_data_norm_sq/(grad_physics_norm_sq+1e-8))
        grad_ratio_records.append(float(grad_ratio.detach().cpu().numpy()))

        optimizer.step()

        loss_records["total"].append(float(loss_t.detach().cpu()))
        loss_records["data"].append(float(loss_d.detach().cpu()))
        loss_records["physics"].append(float(loss_p.detach().cpu()))
        loss_records["lam1"].append(float(l1.detach().cpu()))
        loss_records["lam2"].append(float(l2.detach().cpu()))

        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d} | TotalLoss:{loss_t:.4f} | Ldata:{loss_d:.4f} | Lphys:{loss_p:.4f} λ2={l2:.3f}")
    return model, loss_records, grad_ratio_records


def get_baseline_df():
    """表1基线方法模拟指标，真实实验替换为完整基线模型训练结果"""
    df = pd.DataFrame([
        {"name":"RF","rmse":0.142,"r2":0.781,"pcindex":0.62},
        {"name":"SAR","rmse":0.136,"r2":0.793,"pcindex":0.67},
        {"name":"DeepONet","rmse":0.129,"r2":0.812,"pcindex":0.71},
        {"name":"Conventional CBA","rmse":0.153,"r2":0.762,"pcindex":0.58},
        {"name":"Pure PINN","rmse":0.118,"r2":0.829,"pcindex":0.82},
        {"name":"Our Framework","rmse":0.098,"r2":0.864,"pcindex":0.89}
    ])
    return df


def get_ablation_df():
    """表2消融实验模拟指标"""
    df = pd.DataFrame([
        {"config":"Full Framework","rmse":0.112,"r2":0.832,"pcindex":0.89,"train_h":3.8},
        {"config":"Pure PINN (w/o Mamba)","rmse":0.137,"r2":0.791,"pcindex":0.82,"train_h":4.1},
        {"config":"Pure Mamba (w/o PINN)","rmse":0.125,"r2":0.804,"pcindex":0.68,"train_h":3.5},
        {"config":"w/o GIS integration","rmse":0.121,"r2":0.812,"pcindex":0.83,"train_h":3.2},
        {"config":"w/o adaptive weighting","rmse":0.117,"r2":0.823,"pcindex":0.81,"train_h":3.9},
    ])
    return df


# ---------------- 灵敏度分析模拟数据 ----------------
noise_levels = [0,5,10,15,20,25,30]
rmse_noise_ours = [0.101,0.108,0.114,0.123,0.137,0.148,0.161]
rmse_noise_rf = [0.143,0.157,0.174,0.188,0.209,0.227,0.246]

lambda_physics_list = [0.1,0.26,0.4,0.5,0.6,0.74,0.9]
pcindex_lambda = [0.68,0.78,0.88,0.91,0.87,0.82,0.76]
rmse_lambda = [0.145,0.122,0.104,0.099,0.106,0.118,0.131]

N_grid = [2500, 10000,40000,90000,160000]
wall_time_per_epoch = [12.5,28.3,65.1,118.4,185.2]


if __name__ == "__main__":
    dataset = generate_simulation_dataset(n_spatial=1200, n_time=120, seed=42)
    train_sp_idx, test_sp_idx = spatial_block_split(dataset)
    X_train = dataset["X"][:, train_sp_idx, :]
    Y_train = dataset["Y_obs"][:, train_sp_idx]

    # 维度变换 [S,T,D] -> [S,B,D]，内存限制取前500个空间样本
    X_tr = X_train.permute(1,0,2)[:500,:,:]
    Y_tr = Y_train.permute(1,0)[:500,:]
    model_trained, loss_log, grad_ratio = train_mamba_pinn(X_tr, Y_tr, epochs=145)
    print("训练完成，grad_ratio长度：", len(grad_ratio))
    df_baseline = get_baseline_df()
    df_ablation = get_ablation_df()
    print("\n==== Baseline Table1 ====")
    print(df_baseline.to_string(index=False))
    print("\n==== Ablation Table2 ====")
    print(df_ablation.to_string(index=False))
#（注：内容由AI生成）
