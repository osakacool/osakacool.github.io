import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNBranch(nn.Module):
    """CNN分支：提取单个食品内部局部营养特征；论文公式(3)"""
    def __init__(self, in_dim=128, num_filter=64, kernel_width=3, out_dim=128):
        super().__init__()
        self.conv1d = nn.Conv1d(in_channels=1, out_channels=num_filter, kernel_size=kernel_width, padding=1)
        self.maxpool = nn.AdaptiveMaxPool1d(output_size=1)
        self.proj = nn.Linear(num_filter, out_dim)

    def forward(self, x):
        # x: [B, food_nut_dim]
        B,D = x.shape
        x = x.unsqueeze(1) # B,1,D
        c = F.relu(self.conv1d(x)) # B,64,D
        pool_out = self.maxpool(c).squeeze(-1) # B,64
        return self.proj(pool_out) # B, out_dim


class LSTMBranch(nn.Module):
    """LSTM分支：多天膳食时序依赖，论文2.2.2"""
    def __init__(self, input_dim=128, hidden_dim=128, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)

    def forward(self, seq_x):
        # seq_x [B,T,D] T为膳食序列长度
        out,(h_n,_)=self.lstm(seq_x)
        return h_n[-1,:,:] # B, hidden_dim


class TransformerBranch(nn.Module):
    """Transformer多头自注意力，公式(4)"""
    def __init__(self, feat_dim=128, n_heads=4, n_encoder_layers=2):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=feat_dim, nhead=n_heads,batch_first=True,dim_feedforward=256)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_encoder_layers)

    def forward(self, plan_seq):
        # plan_seq [B,T,D]
        enc_out = self.encoder(plan_seq)
        return torch.mean(enc_out,dim=1) #全局聚合 B,D


class SportProfileEncoder(nn.Module):
    """运动员profile编码：运动类型embedding +体重、训练负荷归一投影"""
    def __init__(self, n_sport_types=8, emb_size=16, out_proj_dim=256):
        super().__init__()
        self.sport_emb = nn.Embedding(n_sport_types, emb_size)
        self.proj_cont = nn.Linear(2, emb_size) #体重、训练负荷两个连续变量
        self.merge_proj = nn.Linear(emb_size*2, out_proj_dim)

    def forward(self, sport_type_idx, weight, training_load):
        emb_cat = self.sport_emb(sport_type_idx) # B,16
        cont_feat = torch.stack([weight, training_load],dim=-1) # B,2
        emb_cont = self.proj_cont(cont_feat) # B,16
        merge = torch.cat([emb_cat, emb_cont], dim=-1)
        return self.merge_proj(merge) # B,256


class TriModelEnsemble(nn.Module):
    """完整CNN‑LSTM‑Transformer三模型集成；图3a/b"""
    def __init__(
        self, nut_dim=128, cnn_out=128, lstm_hidden=128, trans_dim=128,
        sport_emb_out=256, vocab_size=8215 #食品总词汇大小等于融合数据集记录数
    ):
        super().__init__()
        self.cnn_branch = CNNBranch(in_dim=nut_dim, num_filter=64, kernel_width=3, out_dim=cnn_out)
        self.lstm_branch = LSTMBranch(input_dim=nut_dim, hidden_dim=lstm_hidden, num_layers=1)
        self.trans_branch = TransformerBranch(feat_dim=trans_dim, n_heads=4, n_encoder_layers=2)

        concat_total_dim = cnn_out + lstm_hidden + trans_dim
        self.concat_proj = nn.Linear(concat_total_dim, 256)

        self.profile_encoder = SportProfileEncoder(n_sport_types=8, emb_size=16, out_proj_dim=256)

        # condition后解码器输出食品词汇分布
        decode_in_dim = 256+256
        self.decoder = nn.Sequential(
            nn.Linear(decode_in_dim,512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, vocab_size)
        )

    def forward(self, single_food_feat, multi_day_seq, meal_plan_seq, sport_idx, weight, load):
        """
        single_food_feat: B,D 单食品特征，给CNN
        multi_day_seq: B,T,D 多天序列，给LSTM
        meal_plan_seq: B,T2,D 一餐内食品序列，给Transformer
        sport_idx: B 运动类别index
        weight: B 体重kg归一
        load: B训练负荷归一
        return logits食品预测分布，latent_z
        """
        f_cnn = self.cnn_branch(single_food_feat)
        f_lstm = self.lstm_branch(multi_day_seq)
        f_trans = self.trans_branch(meal_plan_seq)

        z = torch.cat([f_cnn,f_lstm,f_trans],dim=-1)
        z = self.concat_proj(z) # B,256

        e_u = self.profile_encoder(sport_idx, weight, load) #B,256
        z_u = torch.cat([z,e_u],dim=-1) #条件表示 Zu=[Z;eu]

        logits = self.decoder(z_u)
        return logits, z_u


def constraint_aware_loss(logits, target_tokens, nut_violation, lambda_penalty=1.0):
    """论文2.2.3约束感知损失 L = Lce + Σ λ_i ϕ_i(P̃)
    nut_violation: [B]每条样本营养违背值
    """
    ce_loss = F.cross_entropy(logits, target_tokens)
    penalty_term = lambda_penalty * torch.mean(nut_violation)
    total_loss = ce_loss + penalty_term
    return total_loss, ce_loss, penalty_term
