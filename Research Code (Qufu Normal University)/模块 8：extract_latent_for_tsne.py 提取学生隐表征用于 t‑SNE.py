# extract_latent_for_tsne.py
import torch
import numpy as np
from liquid_mamba_model import LiquidMamba

@torch.no_grad()
def extract_latent_embedding(model_path, dataloader, device="cuda"):
    """
    加载训练好的Liquid‑Mamba，提取每个学生样本聚合后latent z，用于t‑SNE可视化
    return: emb_np [N, latent_dim], label_bin_np [N,3]
    """
    ckpt = torch.load(model_path, map_location=device)
    model = LiquidMamba().to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    emb_list = []
    label_list = []
    for batch_x, batch_y in dataloader:
        batch_x = batch_x.to(device)
        # forward到mean‑pool之后，跳过分类头
        H = model.ltc(batch_x)
        H = model.drop1(H)
        Y = model.mamba_blocks(H)
        Y = model.drop2(Y)
        z = torch.mean(Y, dim=1)
        emb_list.append(z.cpu().numpy())
        label_list.append(batch_y.numpy())
    emb_np = np.concatenate(emb_list, axis=0)
    label_bin_np = np.concatenate(label_list, axis=0)
    return emb_np, label_bin_np


if __name__ == "__main__":
    # emb, labels = extract_latent_embedding("./best_liquidmamba.pt", val_loader)
    # np.savez("latent_tsne_data.npz", emb=emb, label_bin=labels)
    pass
