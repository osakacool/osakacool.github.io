import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linprog
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from scipy.spatial.distance import jaccard

# ====================== 通用评估工具函数（复用metrics逻辑） ======================
def nutritional_adequacy(generated_nut: np.ndarray, ref_nut: np.ndarray, tol_ratio=0.10):
    """NA 营养充足率 ±10%容忍"""
    lower = ref_nut * (1 - tol_ratio)
    upper = ref_nut * (1 + tol_ratio)
    ok_mask = np.logical_and(generated_nut >= lower, generated_nut <= upper)
    return float(np.mean(ok_mask)) * 100


def constraint_satisfaction_rate(generated_nut: np.ndarray, constraint_upper: list):
    """CSR 约束满足率，constraint_upper是每条样本的上限数组"""
    n = generated_nut.shape[0]
    ok_count = 0
    for i in range(n):
        valid = True
        for dim_idx, limit in enumerate(constraint_upper[i]):
            if generated_nut[i, dim_idx] > limit:
                valid = False
        if valid:
            ok_count += 1
    return ok_count / n * 100


def plan_diversity(food_sets_list):
    """PD 膳食多样性，集合间Jaccard距离均值"""
    dists = []
    n = len(food_sets_list)
    for i in range(n):
        for j in range(i + 1, n):
            s1 = set(food_sets_list[i])
            s2 = set(food_sets_list[j])
            union = len(s1 | s2)
            inter = len(s1 & s2)
            if union == 0:
                dist = 0.0
            else:
                dist = 1.0 - (inter / union)
            dists.append(dist)
    return float(np.mean(dists)) if len(dists) > 0 else 0.0


def generation_fidelity_f1(pred_cat, ref_cat):
    """GF macro‑F1"""
    return f1_score(ref_cat, pred_cat, average="macro")


# =============================================================================
# Baseline1: Linear Programming 线性规划 Simplex
# 目标：逼近目标营养素，求解最优食品份量；论文：固定营养目标优化范式
# =============================================================================
class LinearProgrammingBaseline:
    def __init__(self, food_nutrient_matrix: np.ndarray):
        """
        food_nutrient_matrix: [n_food, n_nut] 融合数据集，每个食品的营养向量
        """
        self.food_mat = food_nutrient_matrix
        self.n_food = food_nutrient_matrix.shape[0]

    def generate_plan(self, target_nutrient: np.ndarray, nut_upper_bounds: np.ndarray):
        """
        target_nutrient: (n_nut,) 参考营养目标
        nut_upper_bounds: (n_nut,) 营养素上限约束
        返回：food_weights(食品份量), plan_nut(生成方案总营养)
        """
        n_nut = len(target_nutrient)
        # 目标：最小化与target偏差，linprog只支持min
        c = np.zeros(self.n_food + n_nut)
        c[self.n_food:] = 1.0

        # A_ub @ x <= b_ub
        A_ub = []
        b_ub = []
        # 约束1：总营养 <= upper bound
        for nut_idx in range(n_nut):
            row = np.zeros(self.n_food + n_nut)
            row[:self.n_food] = self.food_mat[:, nut_idx]
            A_ub.append(row)
            b_ub.append(nut_upper_bounds[nut_idx])

        # 约束2：绝对值松弛变量 |sum‑target| <= slack
        for nut_idx in range(n_nut):
            row_pos = np.zeros(self.n_food + n_nut)
            row_pos[:self.n_food] = self.food_mat[:, nut_idx]
            row_pos[self.n_food + nut_idx] = -1.0
            A_ub.append(row_pos)
            b_ub.append(target_nutrient[nut_idx])

            row_neg = np.zeros(self.n_food + n_nut)
            row_neg[:self.n_food] = -self.food_mat[:, nut_idx]
            row_neg[self.n_food + nut_idx] = -1.0
            A_ub.append(row_neg)
            b_ub.append(-target_nutrient[nut_idx])

        bounds = [(0, None) for _ in range(self.n_food + n_nut)]
        res = linprog(c, A_ub=np.array(A_ub), b_ub=np.array(b_ub), bounds=bounds, method="highs")
        if not res.success:
            food_w = np.zeros(self.n_food)
        else:
            food_w = res.x[:self.n_food]
        plan_nut = np.sum(self.food_mat * food_w.reshape(-1, 1), axis=0)
        return food_w, plan_nut

    def batch_predict(self, target_nut_batch, upper_batch):
        """批量生成"""
        out_nut_list = []
        food_set_list = []
        for t, u in zip(target_nut_batch, upper_batch):
            w, p_nut = self.generate_plan(t, u)
            out_nut_list.append(p_nut)
            selected_food_idx = np.nonzero(w > 1e‑4)[0].tolist()
            food_set_list.append(selected_food_idx)
        return np.array(out_nut_list), food_set_list


# =============================================================================
# Baseline2: Random Forest 随机森林贪心膳食组装
# 输入营养特征预测食品类别，贪心迭代挑选食品构建膳食方案
# =============================================================================
class RandomForestDietBaseline:
    def __init__(self, n_estimators=100):
        self.rf = RandomForestClassifier(n_estimators=n_estimators, n_jobs=-1)

    def fit(self, X_train_nut, y_train_food_cat):
        """
        X_train_nut: [N, nut_dim] profile对应的目标营养
        y_train_food_cat: [N] 食品类别标签
        """
        self.rf.fit(X_train_nut, y_train_food_cat)

    def greedy_assemble(self, target_nut, food_pool_nut, max_item=12):
        """贪心迭代选食品直到接近目标营养"""
        selected = []
        current_nut = np.zeros_like(target_nut)
        for _ in range(max_item):
            feat = np.concatenate([target_nut - current_nut]).reshape(1, -1)
            pred_proba = self.rf.predict_proba(feat)[0]
            top_idx = int(np.argmax(pred_proba))
            selected.append(top_idx)
            current_nut += food_pool_nut[top_idx]
        return selected, current_nut

    def batch_predict(self, target_nut_batch, food_pool_nut):
        food_sets = []
        nut_out = []
        for t in target_nut_batch:
            food_idx_list, cur_nut = self.greedy_assemble(t, food_pool_nut)
            food_sets.append(food_idx_list)
            nut_out.append(cur_nut)
        return np.array(nut_out), food_sets


# =============================================================================
# Baseline3: CNN‑only 仅CNN分支模型
# 只用CNN提取单食品局部特征；没有LSTM、Transformer分支
# =============================================================================
class CNNOnlyModel(nn.Module):
    def __init__(self, nut_dim=128, num_filter=64, kernel_width=3, out_hidden=256, vocab_size=8215):
        super().__init__()
        self.conv1d = nn.Conv1d(in_channels=1, out_channels=num_filter, kernel_size=kernel_width, padding=1)
        self.maxpool = nn.AdaptiveMaxPool1d(1)
        self.proj_cnn = nn.Linear(num_filter, out_hidden)

        # 运动员profile编码，和主模型保持一致
        self.sport_emb = nn.Embedding(8, 16)
        self.proj_cont = nn.Linear(2, 16)
        self.merge_profile = nn.Linear(32, out_hidden)

        decode_in = out_hidden + out_hidden
        self.decoder = nn.Sequential(
            nn.Linear(decode_in, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, vocab_size)
        )

    def forward(self, single_food_feat, sport_idx, weight, training_load):
        B, D = single_food_feat.shape
        x = single_food_feat.unsqueeze(1)
        c = F.relu(self.conv1d(x))
        pool_out = self.maxpool(c).squeeze(-1)
        z_cnn = self.proj_cnn(pool_out)

        # profile编码
        emb_sport = self.sport_emb(sport_idx)
        cont_in = torch.stack([weight, training_load], dim=-1)
        emb_cont = self.proj_cont(cont_in)
        profile_z = self.merge_profile(torch.cat([emb_sport, emb_cont], dim=-1))

        z_u = torch.cat([z_cnn, profile_z], dim=-1)
        logits = self.decoder(z_u)
        return logits


# =============================================================================
# Baseline4: LSTM‑only 仅LSTM分支自回归生成膳食序列
# =============================================================================
class LSTMOnlyModel(nn.Module):
    def __init__(self, nut_dim=128, hidden_dim=128, n_layers=1, vocab_size=8215):
        super().__init__()
        self.lstm = nn.LSTM(nut_dim, hidden_dim, num_layers=n_layers, batch_first=True)

        self.sport_emb = nn.Embedding(8, 16)
        self.proj_cont = nn.Linear(2, 16)
        self.merge_profile = nn.Linear(32, hidden_dim)

        decode_in = hidden_dim + hidden_dim
        self.decoder = nn.Sequential(
            nn.Linear(decode_in, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, vocab_size)
        )

    def forward(self, seq_food_feat, sport_idx, weight, training_load):
        """seq_food_feat: [B, T, D]"""
        out, (h_n, _) = self.lstm(seq_food_feat)
        z_lstm = h_n[-1, :, :]

        emb_sport = self.sport_emb(sport_idx)
        cont_in = torch.stack([weight, training_load], dim=-1)
        emb_cont = self.proj_cont(cont_in)
        profile_z = self.merge_profile(torch.cat([emb_sport, emb_cont], dim=-1))

        z_u = torch.cat([z_lstm, profile_z], dim=-1)
        logits = self.decoder(z_u)
        return logits


# =============================================================================
# Baseline5: Transformer‑only 仅Transformer多头自注意力基线
# 论文最强单分支基线
# =============================================================================
class TransformerOnlyModel(nn.Module):
    def __init__(self, feat_dim=128, n_heads=4, n_encoder_layers=2, vocab_size=8215):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feat_dim, nhead=n_heads, batch_first=True, dim_feedforward=256
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_encoder_layers)

        self.sport_emb = nn.Embedding(8, 16)
        self.proj_cont = nn.Linear(2, 16)
        self.merge_profile = nn.Linear(32, feat_dim)

        decode_in = feat_dim + feat_dim
        self.decoder = nn.Sequential(
            nn.Linear(decode_in, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, vocab_size)
        )

    def forward(self, meal_seq_feat, sport_idx, weight, training_load):
        """meal_seq_feat: [B, T, D]"""
        enc_out = self.encoder(meal_seq_feat)
        z_trans = torch.mean(enc_out, dim=1)

        emb_sport = self.sport_emb(sport_idx)
        cont_in = torch.stack([weight, training_load], dim=-1)
        emb_cont = self.proj_cont(cont_in)
        profile_z = self.merge_profile(torch.cat([emb_sport, emb_cont], dim=-1))

        z_u = torch.cat([z_trans, profile_z], dim=-1)
        logits = self.decoder(z_u)
        return logits


# =============================================================================
# 统一评估入口函数：输入模型/实例、测试集数据，输出四指标 NA,CSR,PD,GF
# =============================================================================
def evaluate_baseline(model, test_data, baseline_type: str, device="cpu"):
    """
    baseline_type选择：["lp", "rf", "cnn_only", "lstm_only", "trans_only"]
    test_data 字典包含：
        target_nut_batch, upper_batch, food_pool_nut,
        single_food_feat, seq_feat, meal_seq_feat,
        sport_idx, weight, load, ref_food_cat
    """
    td = test_data
    if baseline_type == "lp":
        lp_model = model
        gen_nut, food_sets = lp_model.batch_predict(td["target_nut_batch"], td["upper_batch"])
    elif baseline_type == "rf":
        rf_model = model
        gen_nut, food_sets = rf_model.batch_predict(td["target_nut_batch"], td["food_pool_nut"])
    elif baseline_type == "cnn_only":
        model.eval()
        sf = torch.from_numpy(td["single_food_feat"]).to(device)
        sp = torch.from_numpy(td["sport_idx"]).to(device)
        w = torch.from_numpy(td["weight"]).to(device)
        ld = torch.from_numpy(td["load"]).to(device)
        with torch.no_grad():
            logits = model(sf, sp, w, ld)
        pred_tokens = torch.argmax(logits, dim=-1).cpu().numpy()
        food_sets = [[int(t)] for t in pred_tokens]
        gen_nut = td["food_pool_nut"][pred_tokens]
    elif baseline_type == "lstm_only":
        model.eval()
        sq = torch.from_numpy(td["seq_feat"]).to(device)
        sp = torch.from_numpy(td["sport_idx"]).to(device)
        w = torch.from_numpy(td["weight"]).to(device)
        ld = torch.from_numpy(td["load"]).to(device)
        with torch.no_grad():
            logits = model(sq, sp, w, ld)
        pred_tokens = torch.argmax(logits, dim=-1).cpu().numpy()
        food_sets = [[int(t)] for t in pred_tokens]
        gen_nut = td["food_pool_nut"][pred_tokens]
    elif baseline_type == "trans_only":
        model.eval()
        ms = torch.from_numpy(td["meal_seq_feat"]).to(device)
        sp = torch.from_numpy(td["sport_idx"]).to(device)
        w = torch.from_numpy(td["weight"]).to(device)
        ld = torch.from_numpy(td["load"]).to(device)
        with torch.no_grad():
            logits = model(ms, sp, w, ld)
        pred_tokens = torch.argmax(logits, dim=-1).cpu().numpy()
        food_sets = [[int(t)] for t in pred_tokens]
        gen_nut = td["food_pool_nut"][pred_tokens]
    else:
        raise ValueError("unknown baseline type")

    na = nutritional_adequacy(gen_nut, td["target_nut_batch"])
    csr = constraint_satisfaction_rate(gen_nut, td["upper_batch"])
    pd = plan_diversity(food_sets)
    gf = generation_fidelity_f1([fs[0] for fs in food_sets], td["ref_food_cat"])
    return {"NA": na, "CSR": csr, "PD": pd, "GF": gf}


# ============================ 仿真测试入口 =============================
if __name__ == "__main__":
    np.random.seed(42)
    torch.manual_seed(42)
    n_test = 200
    nut_dim = 128
    n_food = 8215

    food_pool_nut = np.random.randn(n_food, nut_dim)
    target_nut_batch = np.random.randn(n_test, nut_dim)
    upper_batch = np.tile(np.array([3000, 120, 400, 2500, 150]), (n_test, 1))
    ref_food_cat = np.random.randint(0, n_food, size=n_test)

    test_dict = {
        "target_nut_batch": target_nut_batch,
        "upper_batch": upper_batch,
        "food_pool_nut": food_pool_nut,
        "single_food_feat": np.random.randn(n_test, nut_dim),
        "seq_feat": np.random.randn(n_test, 7, nut_dim),
        "meal_seq_feat": np.random.randn(n_test, 10, nut_dim),
        "sport_idx": np.random.randint(0, 8, n_test),
        "weight": np.random.uniform(50, 110, n_test),
        "load": np.random.choice([0, 1, 2], n_test),
        "ref_food_cat": ref_food_cat
    }

    # 1 LP基线
    lp = LinearProgrammingBaseline(food_pool_nut)
    res_lp = evaluate_baseline(lp, test_dict, baseline_type="lp")
    print(f"[LinearProgramming] {res_lp}")

    # 2 RF基线
    rf = RandomForestDietBaseline()
    X_rf = target_nut_batch
    y_rf = ref_food_cat
    rf.fit(X_rf, y_rf)
    res_rf = evaluate_baseline(rf, test_dict, baseline_type="rf")
    print(f"[RandomForest] {res_rf}")

    # 3 CNN‑only
    cnn_only = CNNOnlyModel(nut_dim=128, vocab_size=n_food)
    res_cnn = evaluate_baseline(cnn_only, test_dict, baseline_type="cnn_only")
    print(f"[CNN‑only] {res_cnn}")

    #4 LSTM‑only
    lstm_only = LSTMOnlyModel(nut_dim=128, vocab_size=n_food)
    res_lstm = evaluate_baseline(lstm_only, test_dict, baseline_type="lstm_only")
    print(f"[LSTM‑only] {res_lstm}")

    #5 Transformer‑only
    trans_only = TransformerOnlyModel(feat_dim=128, vocab_size=n_food)
    res_trans = evaluate_baseline(trans_only, test_dict, baseline_type="trans_only")
    print(f"[Transformer‑only] {res_trans}")
