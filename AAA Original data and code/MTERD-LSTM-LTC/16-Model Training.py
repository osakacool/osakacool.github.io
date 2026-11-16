"""Training utilities; fit on training participants and monitor validation participants."""
import copy
import torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset

def train_model(model,x_train,y_train,x_val,y_val,epochs=100,batch_size=32,lr=0.001,patience=10,device='cpu'):
    model=model.to(device);opt=torch.optim.Adam(model.parameters(),lr=lr)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode='min',factor=0.1,patience=5)
    ds=TensorDataset(torch.as_tensor(x_train,dtype=torch.float32),torch.as_tensor(y_train,dtype=torch.float32))
    loader=DataLoader(ds,batch_size=batch_size,shuffle=True)
    vx=torch.as_tensor(x_val,dtype=torch.float32,device=device);vy=torch.as_tensor(y_val,dtype=torch.float32,device=device)
    best=float('inf');best_weights=None;stale=0
    for epoch in range(epochs):
        model.train()
        for x,y in loader:
            x=x.to(device);y=y.to(device);opt.zero_grad()
            loss=nn.functional.mse_loss(model(x),y);loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step()
        model.eval()
        with torch.no_grad():val=float(nn.functional.mse_loss(model(vx),vy))
        scheduler.step(val)
        if val<best-1e-7:best=val;best_weights=copy.deepcopy(model.state_dict());stale=0
        else:stale+=1
        if stale>=patience:break
    model.load_state_dict(best_weights)
    return model,{'best_validation_mse':best,'epochs_completed':epoch+1}
