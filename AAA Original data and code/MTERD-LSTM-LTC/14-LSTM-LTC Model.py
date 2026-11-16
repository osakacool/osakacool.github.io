"""Input-dependent time-constant gated LSTM implementation in PyTorch.

This is a documented operational implementation of the manuscript's described
mechanism; exact equivalence requires the authors' original equation-level code.
"""
import torch
from torch import nn

class LiquidLSTMCell(nn.Module):
    def __init__(self,input_size,hidden_size,tau_min=0.1,tau_max=30.0,dt=0.1):
        super().__init__();self.hidden_size=hidden_size
        self.gates=nn.Linear(input_size+hidden_size,4*hidden_size)
        self.time_constants=nn.Linear(input_size+hidden_size,3*hidden_size)
        self.tau_min=tau_min;self.tau_max=tau_max;self.dt=dt
    def forward(self,x,state):
        h,c=state;z=torch.cat([x,h],dim=-1)
        i,f,g,o=self.gates(z).chunk(4,dim=-1)
        tau=self.tau_min+(self.tau_max-self.tau_min)*torch.sigmoid(self.time_constants(z))
        ti,tf,to=tau.chunk(3,dim=-1)
        ai=1-torch.exp(-self.dt/ti);af=1-torch.exp(-self.dt/tf);ao=1-torch.exp(-self.dt/to)
        i=torch.sigmoid(i)*ai;f=1-(1-torch.sigmoid(f))*af;o=torch.sigmoid(o)*ao
        c=f*c+i*torch.tanh(g);h=o*torch.tanh(c)
        return (h,c),tau

class LSTMLTC(nn.Module):
    def __init__(self,input_size=128,hidden_size=128,num_layers=2,dropout=0.2):
        super().__init__();self.cells=nn.ModuleList([LiquidLSTMCell(input_size if j==0 else hidden_size,hidden_size) for j in range(num_layers)])
        self.norms=nn.ModuleList([nn.LayerNorm(input_size if j==0 else hidden_size) for j in range(num_layers)])
        self.dropout=nn.Dropout(dropout);self.head=nn.Linear(hidden_size,1)
    def forward(self,x,return_taus=False):
        if x.ndim!=3:raise ValueError('Expected [batch,time,features]')
        batch,length,_=x.shape;states=[(x.new_zeros(batch,c.hidden_size),x.new_zeros(batch,c.hidden_size)) for c in self.cells]
        history=[];pred=[]
        for t in range(length):
            z=x[:,t,:];taus=[]
            for j,cell in enumerate(self.cells):
                states[j],tau=cell(self.norms[j](z),states[j]);z=states[j][0];taus.append(tau)
                if j<len(self.cells)-1:z=self.dropout(z)
            pred.append(self.head(z));history.append(torch.stack(taus,dim=1))
        y=torch.stack(pred,dim=1).squeeze(-1)
        return (y,torch.stack(history,dim=1)) if return_taus else y
