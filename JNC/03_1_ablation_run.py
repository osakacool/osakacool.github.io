"""
03_1_ablation_run.py
对应论文章节：6.5 Ablation Study
功能：完整5组消融配置训练
配置列表：
1) Full Framework: Mamba + PINN + Adaptive λ + GIS特征
2) Pure PINN w/o Mamba: 移除Mamba编码器，只用MLP做特征映射
3) Pure Mamba w/o PINN: 移除PINN物理损失，只使用Mamba+MSE数据损失
4) w/o GIS integration: 剔除GIS空间坐标特征(x,y)
5) w/o adaptive weighting: λ1 λ2固定常数，不做可学习自适应参数

输出：ablation_result.csv，包含RMSE, R², PC‑Index, 训练耗时
依赖：01_sim_dataset.py，02_model_def.py
"""
import time
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from 01_sim_dataset import generate_simulation_dataset, spatial_block_split
from 02_model_def import MambaSpatialTemporalEncoder, PINNValuationHead, compute_physics_pde_residual

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------
# 消融变体模型定义（5种消融配置）
# --------------------------

# 配置1 Full Framework（完整模型，同02_model_def）
class FullFramework(nn.Module):
    def __init__(self, feat_dim):
        super().__init__()
        self.encoder = MambaSpatialTemporalEncoder(feat_dim=feat_dim, hidden_dim=128)
        self.pinn_head = PINNValuationHead(hidden_dim=128)
        self.lambda1 = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.lambda2 = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))

    def forward(self, X_input):
        latent = self.encoder(X_input)
        pred_val = self.pinn_head(latent)
        return pred_val

# 配置2 Pure PINN w/o Mamba，去掉Mamba，替换普通MLP编码器
class PurePINN_NoMamba(nn.Module):
    def __init__(self, feat_dim):
        super().__init__()
        self.encoder_mlp = nn.Sequential(
            nn.Linear(feat_dim,128), nn.Tanh(),
            nn.Linear(128,128), nn.Tanh(),
        )
        self.pinn_head = PINNValuationHead(hidden_dim=128)
        self.lambda1 = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.lambda2 = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))

    def forward(self, X_input):
        latent = self.encoder_mlp(X_input)
        pred_val = self.pinn_head(latent)
        return pred_val

# 配置3 Pure Mamba w/o PINN，移除物理损失，只有数据MSE损失
class PureMamba_NoPINN(nn.Module):
    def __init__(self, feat_dim):
        super().__init__()
        self.encoder = MambaSpatialTemporalEncoder(feat_dim=feat_dim, hidden_dim=128)
        self.pinn_head = PINNValuationHead(hidden_dim=128)

    def forward(self, X_input):
        latent = self.encoder(X_input)
        pred_val = self.pinn_head(latent)
        return pred_val

# 配置5 w/o adaptive weighting，权重固定不可学习
class MambaPINN_FixedLambda(nn.Module):
    def __init__(self, feat_dim, lam1_fixed=1.0, lam2_fixed=0.1):
        super().__init__()
        self.encoder = MambaSpatialTemporalEncoder(feat_dim=feat_dim, hidden_dim=128)
        self.pinn_head = PINNValuationHead(hidden_dim=128)
        self.lam1_fixed = lam1_fixed
        self.lam2_fixed = lam2_fixed

    def forward(self, X_input):
        latent = self.encoder(X_input)
        pred_val = self.pinn_head(latent)
        return pred_val

# --------------------------
# 损失函数工厂，适配不同消融配置
# --------------------------
def loss_ablation(model, cfg_type, X_batch, Y_batch, collocation_t, collocation_x, collocation_y):
    pred = model(X_batch)
    loss_data = torch.mean((pred - Y_batch)**2)

    # Pure Mamba w/o PINN: 无物理损失，直接返回MSE
    if cfg_type == "pure_mamba_no_pinn":
        return loss_data, loss_data, torch.tensor(0.0), None, None

    # 计算PDE物理残差
    N_col = collocation_t.shape[0]
    X_col = torch.randn(N_col,1,X_batch.shape[-1], device=device, requires_grad=True)
    if hasattr(model, "encoder"):
        latent_col = model.encoder(X_col)
    elif hasattr(model, "encoder_mlp"):
        latent_col = model.encoder_mlp(X_col)
    V_col = model.pinn_head(latent_col[:,0,:])
    pde_res = compute_physics_pde_residual(V_col, collocation_t, collocation_x, collocation_y)
    loss_physics = torch.mean(pde_res**2)

    if cfg_type == "fixed_lambda":
        lam1 = torch.tensor(model.lam1_fixed)
        lam2 = torch.tensor(model.lam2_fixed)
        loss_total = lam1 * loss_data + lam2 * loss_physics
    else:
        lam1 = torch.clamp(model.lambda1, 0.01,2.0)
        lam2 = torch.clamp(model.lambda2, 0.01,0.9)
        loss_total = lam1 * loss_data + lam2 * loss_physics

    return loss_total, loss_data, loss_physics, lam1, lam2


def calc_metrics(y_true:torch.Tensor, y_pred:torch.Tensor, loss_physics:float):
    """
    计算评估指标 RMSE, R², PC‑Index(论文公式39‑40简化实现)
    PC‑Index ∈ [0,1]，物理约束越满足越接近1
    """
    yt = y_true.detach().cpu().numpy().reshape(-1)
    yp = y_pred.detach().cpu().numpy().reshape(-1)
    rmse = float(torch.sqrt(torch.mean((y_true-y_pred)**2)).cpu().numpy())

    ss_res = ((yt - yp)**2).sum()
    ss_tot = ((yt - yt.mean())**2).sum()
    r2 = 1.0 - (ss_res / ss_tot)

    # PC‑Index 简化实现，对应论文exp(-β*violation)
    beta = 2.2
    pc_index = float(np.exp(-beta * loss_physics))
    pc_index = np.clip(pc_index, 0.0,1.0)
    return {"rmse":rmse, "r2":r2, "pcindex":pc_index}


def run_single_ablation(cfg_name, cfg_type, X_train, Y_train, X_test, Y_test, epochs=100):
    """执行单组消融实验，返回指标+耗时"""
    t0 = time.time()
    n_feat = X_train.shape[-1]
    # 初始化模型
    if cfg_type == "full":
        model = FullFramework(n_feat).to(device)
    elif cfg_type == "pure_pinn_no_mamba":
        model = PurePINN_NoMamba(n_feat).to(device)
    elif cfg_type == "pure_mamba_no_pinn":
        model = PureMamba_NoPINN(n_feat).to(device)
    elif cfg_type == "fixed_lambda":
        model = MambaPINN_FixedLambda(n_feat, lam1_fixed=1.0, lam2_fixed=0.1).to(device)
    else:
        raise ValueError("unknown cfg_type")

    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
    n_collocation = 8000
    collocation_t = torch.rand(n_collocation, device=device, requires_grad=True)
    collocation_x = torch.rand(n_collocation, device=device, requires_grad=True)
    collocation_y = torch.rand(n_collocation, device=device, requires_grad=True)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        loss_t, loss_d, loss_p, _, _ = loss_ablation(
            model, cfg_type, X_train, Y_train, collocation_t, collocation_x, collocation_y
        )
        loss_t.backward()
        optimizer.step()
        if epoch%20==0:
            print(f"[{cfg_name:25s}] epoch {epoch:3d} total_loss={float(loss_t):.4f}")

    train_elapsed_h = (time.time()-t0)/3600.0
    # 测试集评估
    model.eval()
    with torch.no_grad():
        y_pred_test = model(X_test)
    # 获取测试集物理loss
    _, _, loss_p_test, _, _ = loss_ablation(model, cfg_type, X_test, Y_test, collocation_t, collocation_x, collocation_y)
    met = calc_metrics(Y_test, y_pred_test, float(loss_p_test.cpu()))
    met["config_name"] = cfg_name
    met["train_h"] = round(train_elapsed_h,3)
    return met


if __name__ == "__main__":
    # 加载模拟数据集
    dataset = generate_simulation_dataset(n_spatial=1000, n_time=100, seed=42)
    tr_idx, te_idx = spatial_block_split(dataset)

    X_all = dataset["X"]
    Y_all = dataset["Y_obs"]
    X_train_raw = X_all[:,tr_idx,:]
    Y_train_raw = Y_all[:,tr_idx]
    X_test_raw  = X_all[:,te_idx,:]
    Y_test_raw  = Y_all[:,te_idx]

    # ------------------------------
    # 消融4：w/o GIS integration，删除最后两列GIS坐标特征(x,y)
    # ------------------------------
    X_train_no_gis = X_train_raw[:,:,0:-2]
    X_test_no_gis  = X_test_raw[:,:,0:-2]
    # 维度变换 [T,S,D] → [S,T,D] (batch=spatial sample)
    X_tr = X_train_raw.permute(1,0,2)[:400,:,:]
    Y_tr = Y_train_raw.permute(1,0)[:400,:]
    X_te = X_test_raw.permute(1,0,2)[:120,:,:]
    Y_te = Y_test_raw.permute(1,0)[:120,:]

    X_tr_no_gis = X_train_no_gis.permute(1,0,2)[:400,:,:]
    X_te_no_gis = X_test_no_gis.permute(1,0,2)[:120,:,:]

    ablation_configs = [
        {"name":"Full Framework", "type":"full", "X_tr":X_tr, "X_te":X_te},
        {"name":"Pure PINN (w/o Mamba)", "type":"pure_pinn_no_mamba", "X_tr":X_tr, "X_te":X_te},
        {"name":"Pure Mamba (w/o PINN)", "type":"pure_mamba_no_pinn", "X_tr":X_tr, "X_te":X_te},
        {"name":"w/o GIS integration", "type":"full", "X_tr":X_tr_no_gis, "X_te":X_te_no_gis},
        {"name":"w/o adaptive weighting", "type":"fixed_lambda", "X_tr":X_tr, "X_te":X_te},
    ]

    result_list = []
    for cfg in ablation_configs:
        res = run_single_ablation(
            cfg_name=cfg["name"],
            cfg_type=cfg["type"],
            X_train=cfg["X_tr"],
            Y_train=Y_tr,
            X_test=cfg["X_te"],
            Y_test=Y_te,
            epochs=100
        )
        result_list.append(res)

    # 输出csv结果文件
    out_csv = "ablation_result.csv"
    fieldnames = ["config_name","rmse","r2","pcindex","train_h"]
    with open(out_csv,"w",newline="",encoding="utf‑8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(result_list)
    print(f"\n消融实验全部完成，结果保存在 {out_csv}")
    for r in result_list:
        print(f"{r['config_name']:25s} | RMSE={r['rmse']:.4f} | R2={r['r2']:.4f} | PC‑Index={r['pcindex']:.4f} | time(h)={r['train_h']:.3f}")
#（注：内容由AI生成）
