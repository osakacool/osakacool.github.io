# multi_objective_loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class CumulativeLinkLoss(nn.Module):
    """序数回归损失，用于1‑5分序数标签"""
    def __init__(self, num_classes=5):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, logits, y_true):
        """y_true: 0‑4 (original score 1‑5 → minus 1)"""
        y = y_true.long()
        cum_prob = torch.sigmoid(logits)
        loss = 0
        for k in range(self.num_classes-1):
            mask = (y > k).float()
            loss += F.binary_cross_entropy(cum_prob[...,k], mask, reduction="mean")
        return loss


class MultiTaskSELHead(nn.Module):
    """多任务输出头：binary BCE / ordinal / continuous regression"""
    def __init__(self, latent_dim: int):
        super().__init__()
        # binary classification head (main)
        self.head_bin = nn.Linear(latent_dim,3)
        # ordinal regression head (aux) 5‑point scale
        self.head_ord = nn.Linear(latent_dim,4)
        # continuous regression head (aux)
        self.head_cont = nn.Linear(latent_dim,3)

    def forward(self, z_pooled):
        logit_bin = self.head_bin(z_pooled)
        prob_bin = torch.sigmoid(logit_bin)
        logit_ord = self.head_ord(z_pooled)
        pred_cont = self.head_cont(z_pooled)
        return {"prob_bin":prob_bin, "logit_ord":logit_ord, "pred_cont":pred_cont}


def calc_multitask_loss(out_dict, y_bin, y_ord, y_cont, weight_bin=1.0, weight_ord=0.3, weight_cont=0.3):
    loss_bce = F.binary_cross_entropy(out_dict["prob_bin"], y_bin)
    loss_ord = CumulativeLinkLoss()(out_dict["logit_ord"], y_ord)
    loss_mse = F.mse_loss(out_dict["pred_cont"], y_cont)
    total = weight_bin * loss_bce + weight_ord * loss_ord + weight_cont * loss_mse
    return {
        "total_loss": total,
        "loss_bce": loss_bce.item(),
        "loss_ord": loss_ord.item(),
        "loss_mse": loss_mse.item()
    }
