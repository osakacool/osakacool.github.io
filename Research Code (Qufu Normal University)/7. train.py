import torch
import torch.nn as nn
from liquid_mamba_model import LiquidMamba
from metrics import calc_all_metrics
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    model = LiquidMamba(feat_dim=128,ltc_hidden=256,mamba_dim=256).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.9,0.999), weight_decay=1e-5)
    loss_fn = nn.BCELoss()
    max_epoch =100
    batch_size=8
    # ----dataloader placeholder----
    # train_loader,val_loader = get_sem_dataloader()

    for epoch in range(max_epoch):
        model.train()
        total_loss=0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            pred = model(batch_x)
            loss = loss_fn(pred, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        # validation
        model.eval()
        y_trues = []
        y_preds = []
        with torch.no_grad():
            for x_val,y_val in val_loader:
                x_val = x_val.to(device)
                pred_val = model(x_val).cpu().numpy()
                y_trues.append(y_val.numpy())
                y_preds.append(pred_val)
        y_trues = np.concatenate(y_trues,0)
        y_preds = np.concatenate(y_preds,0)
        res = calc_all_metrics(y_trues, y_preds)
        print(f"epoch:{epoch}, macro_auc:{res['macro_auc']:.4f}")

if __name__=="__main__":
    main()
