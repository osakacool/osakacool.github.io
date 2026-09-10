"""
02_model_def.py
对应论文章节：4.1 /4.3 Mamba神经算子、PINN物理约束；公式(6‑30)
功能：Mamba编码器、PINN预测头、PDE残差计算、总损失函数
"""
import torch
import torch.nn as nn
from torch.autograd import grad

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# mamba‑ssm导入容错，如果没有安装使用简易模拟模块
try:
    from mamba_ssm import Mamba
except ImportError:
    class Mamba(nn.Module):
        def __init__(self, d_model, d_state, d_conv, expand):
            super().__init__()
            self.linear_in = nn.Linear(d_model, d_model)
            self.gru = nn.GRU(input_size=d_model, hidden_size=d_model, num_layers=2, batch_first=True)
            self.linear_out = nn.Linear(d_model, d_model)
        def forward(self, x):
            x = self.linear_in(x)
            out, _ = self.gru(x)
            return self.linear_out(out)


class MambaSpatialTemporalEncoder(nn.Module):
    """Mamba时空编码器，论文4.3"""
    def __init__(self, feat_dim:int, hidden_dim=128, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.proj_in = nn.Linear(feat_dim, hidden_dim)
        self.mamba_layer1 = Mamba(d_model=hidden_dim, d_state=d_state, d_conv=d_conv, expand=expand)
        self.mamba_layer2 = Mamba(d_model=hidden_dim, d_state=d_state, d_conv=d_conv, expand=expand)
        self.mamba_layer3 = Mamba(d_model=hidden_dim, d_state=d_state, d_conv=d_conv, expand=expand)
        self.mamba_layer4 = Mamba(d_model=hidden_dim, d_state=d_state, d_conv=d_conv, expand=expand)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        # x: [B, seq_len, feat_dim]
        h = self.proj_in(x)
        h = self.mamba_layer1(h)
        h = self.mamba_layer2(h)
        h = self.mamba_layer3(h)
        h = self.mamba_layer4(h)
        h = self.norm(h)
        return h


class PINNValuationHead(nn.Module):
    """PINN预测输出头，输出归一化估值0‑1"""
    def __init__(self, hidden_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32,1),
            nn.Sigmoid()
        )

    def forward(self, latent_feature):
        return self.mlp(latent_feature).squeeze(-1)


def compute_physics_pde_residual(V_pred, t_coord, x_coord, y_coord):
    """
    碳封存PDE残差，论文公式28 对流‑扩散‑反应方程
    ∂C/∂t + ∇·(C v) = R − D
    """
    V_pred.requires_grad_(True)
    dV_dt = grad(outputs=V_pred.sum(), inputs=t_coord, create_graph=True)[0]
    dV_dx = grad(outputs=V_pred.sum(), inputs=x_coord, create_graph=True)[0]
    dV_dy = grad(outputs=V_pred.sum(), inputs=y_coord, create_graph=True)[0]
    div_term = 0.2*dV_dx + 0.2*dV_dy
    R = 0.15 * torch.ones_like(V_pred)
    D_decay = 0.08 * V_pred
    residual = dV_dt + div_term - (R - D_decay)
    return residual


class MambaPINNFramework(nn.Module):
    """完整Mamba‑PINN双驱动框架，含自适应损失权重λ1 λ2"""
    def __init__(self, feat_dim):
        super().__init__()
        self.encoder = MambaSpatialTemporalEncoder(feat_dim=feat_dim, hidden_dim=128)
        self.pinn_head = PINNValuationHead(hidden_dim=128)
        self.lambda1 = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.lambda2 = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))

    def forward(self, X_input):
        B, T_seq, D_feat = X_input.shape
        latent = self.encoder(X_input)
        pred_val = self.pinn_head(latent)
        return pred_val


def loss_total_calc(model, X_batch, Y_batch, collocation_t, collocation_x, collocation_y):
    """
    总损失函数：公式(8) total_loss = λ1*L_data + λ2*L_physics
    return: loss_total, loss_data, loss_physics, lam1_clamped, lam2_clamped
    """
    pred = model(X_batch)
    loss_data = torch.mean((pred - Y_batch)**2)

    N_col = collocation_t.shape[0]
    X_col = torch.randn(N_col,1,X_batch.shape[-1], device=device, requires_grad=True)
    latent_col = model.encoder(X_col)
    V_col = model.pinn_head(latent_col[:,0,:])
    pde_res = compute_physics_pde_residual(V_col, collocation_t, collocation_x, collocation_y)
    loss_physics = torch.mean(pde_res**2)

    lam1 = torch.clamp(model.lambda1, 0.01, 2.0)
    lam2 = torch.clamp(model.lambda2, 0.01, 0.9)
    loss_total = lam1 * loss_data + lam2 * loss_physics
    return loss_total, loss_data, loss_physics, lam1, lam2
#（注：内容由AI生成）
