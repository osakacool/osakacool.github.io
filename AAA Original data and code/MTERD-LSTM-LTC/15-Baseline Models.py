"""Reference temporal baselines for the manuscript's comparison."""
import torch
from torch import nn
from sklearn.linear_model import Ridge

class StandardLSTM(nn.Module):
    def __init__(self,input_size=128,hidden_size=128):
        super().__init__();self.rnn=nn.LSTM(input_size,hidden_size,batch_first=True,num_layers=2,dropout=0.2);self.head=nn.Linear(hidden_size,1)
    def forward(self,x):return self.head(self.rnn(x)[0]).squeeze(-1)

class AttentionLSTM(nn.Module):
    def __init__(self,input_size=128,hidden_size=128,window=32):
        super().__init__();self.rnn=nn.LSTM(input_size,hidden_size,batch_first=True);self.score=nn.Linear(hidden_size,1);self.head=nn.Linear(hidden_size,1);self.window=window
    def forward(self,x):
        h,_=self.rnn(x);out=[]
        for t in range(h.size(1)):
            local=h[:,max(0,t-self.window+1):t+1,:]
            w=torch.softmax(self.score(local),dim=1)
            out.append(self.head((w*local).sum(dim=1)))
        return torch.stack(out,dim=1).squeeze(-1)

class TCN(nn.Module):
    def __init__(self,input_size=128,hidden_size=128):
        super().__init__();layers=[];in_ch=input_size
        for d in [1,2,4,8,16,32,64,128]:
            layers.extend([nn.Conv1d(in_ch,hidden_size,3,padding=2*d,dilation=d),nn.ReLU()]);in_ch=hidden_size
        self.layers=nn.ModuleList(layers);self.head=nn.Conv1d(hidden_size,1,1)
    def forward(self,x):
        z=x.transpose(1,2);length=z.shape[-1]
        for layer in self.layers:
            z=layer(z)
            if isinstance(layer,nn.Conv1d):z=z[...,:length]
        return self.head(z).squeeze(1)

class PsychoacousticRidge:
    """Fit ridge regression to selected handcrafted acoustic descriptors."""
    def __init__(self,alpha=1.0):self.model=Ridge(alpha=alpha)
    def fit(self,x,y):self.model.fit(x,y);return self
    def predict(self,x):return self.model.predict(x)
