import torch
import torch.nn as nn

class CustomModule(nn.Module):
    def __init__(self,input_dim,hidden_dim,output_dim):
        super().__init__()
        self.layer=nn.Sequential(
            nn.Linear(in_features=input_dim,out_features=hidden_dim),
            nn.ReLU(),
            nn.Linear(in_features=hidden_dim,out_features=output_dim)
        )
    def forward(self,x):
        y=self.layer(x)
        return y

class CustomLinear(nn.Module):
    def __init__(self,input_dim,output_dim):
        super().__init__()
        self.layer=nn.Linear(in_features=input_dim,out_features=output_dim)
    def forward(self,x):
        y=self.layer(x)
        return y