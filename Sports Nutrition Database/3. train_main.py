import torch
import numpy as np
from model import TriModelEnsemble, constraint_aware_loss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 超参 论文2.3.1
N_EPOCH = 120
BATCH_SIZE = 64
LR_INIT = 1e-3
DROPOUT = 0.2
LAMBDA_PENALTY = 1.0
VOCAB_SIZE = 8215

# =========数据集，真实替换为data_fusion输出的真实数据========
def simulate_dataset(n_total=12400):
    """生成的运动员profile+膳食plan对"""
    sport_idx = torch.randint(0,8,(n_total,))
    weight = torch.FloatTensor(np.random.uniform(50,110,size=n_total))
    training_load = torch.FloatTensor(np.random.choice([0.,1.,2.],size=n_total)) #低中高
    single_food_feat = torch.randn(n_total,128)
    multi_day_seq = torch.randn(n_total,7,128) #7天序列
    meal_plan_seq = torch.randn(n_total,10,128) #一餐最多10个食品
    target_tokens = torch.randint(0,VOCAB_SIZE,size=(n_total,))
    nut_violation = torch.abs(torch.randn(n_total)) #营养违背惩罚项
    return sport_idx, weight, training_load, single_food_feat, multi_day_seq, meal_plan_seq, target_tokens, nut_violation

sport_idx_all, weight_all, load_all, sf_feat, md_seq, mp_seq, tgt_tokens, nut_vio = simulate_dataset(12400)

#划分
n_train,n_val,n_test = 8680,1860,1860
train_slice = slice(0,n_train)
val_slice = slice(n_train,n_train+n_val)
test_slice = slice(n_train+n_val,None)

model = TriModelEnsemble(nut_dim=128, vocab_size=VOCAB_SIZE).to(device)
opt = torch.optim.Adam(model.parameters(), lr=LR_INIT)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=5)

# 简单dataloader，正式代码替换为Dataset+DataLoader
def get_batch(slice_obj, batch_i):
    b_start = slice_obj.start + batch_i*BATCH_SIZE
    b_end = min(b_start+BATCH_SIZE, slice_obj.stop)
    return (
        sport_idx_all[b_start:b_end].to(device),
        weight_all[b_start:b_end].to(device),
        load_all[b_start:b_end].to(device),
        sf_feat[b_start:b_end].to(device),
        md_seq[b_start:b_end].to(device),
        mp_seq[b_start:b_end].to(device),
        tgt_tokens[b_start:b_end].to(device),
        nut_vio[b_start:b_end].to(device)
    )

for epoch in range(N_EPOCH):
    model.train()
    total_tr_loss=0.0
    n_batch_train = (n_train+BATCH_SIZE-1)//BATCH_SIZE
    for bi in range(n_batch_train):
        sp_idx,w,ld,sf,md,mp,tgt,nv = get_batch(train_slice, bi)
        logits,zu = model(sf,md,mp,sp_idx,w,ld)
        loss,ce,pen = constraint_aware_loss(logits, tgt, nv, lambda_penalty=LAMBDA_PENALTY)
        opt.zero_grad()
        loss.backward()
        opt.step()
        total_tr_loss += loss.item()

    #验证
    model.eval()
    val_loss_sum=0.0
    n_batch_val=(n_val+BATCH_SIZE-1)//BATCH_SIZE
    with torch.no_grad():
        for bi in range(n_batch_val):
            sp_idx,w,ld,sf,md,mp,tgt,nv = get_batch(val_slice, bi)
            logits,zu = model(sf,md,mp,sp_idx,w,ld)
            loss,_,_ = constraint_aware_loss(logits,tgt,nv,LAMBDA_PENALTY)
            val_loss_sum += loss.item()
    val_avg = val_loss_sum/n_batch_val
    scheduler.step(val_avg)
    print(f"Epoch {epoch:3d} | TrainLoss {total_tr_loss/n_batch_train:.4f} | ValLoss {val_avg:.4f}")

torch.save(model.state_dict(), "trimodel_ensemble.pt")
