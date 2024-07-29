import torch
import torch.nn as nn

class CustomLinear(nn.Module):
    def __init__(self,input_dim,hidden_dim,output_dim):
        self.input_dim=input_dim
        self.hidden_dim=hidden_dim
        self.output_dim=output_dim
    
    def forward(self,x):
        pass
