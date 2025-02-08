import os
import random
import numpy as np
from model import CustomLinear
from dataset import CustomDataset
from model_train import Model_Trainer
import torch
from torch.utils.data import random_split,DataLoader

"""
seed setting
"""
seed=42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed) 
os.environ["PYTHONHASHSEED"]=str(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic=True 
torch.backends.cudnn.benchmark=False

"""
create simple data
"""
x=torch.randn(100,1,dtype=torch.float32)
y=3*x+2
dataset=CustomDataset(x=x,y=y)
train_size=int(0.9*len(dataset))
test_size=len(dataset)-train_size
train_dataset,test_datset=random_split(dataset=dataset,lengths=[train_size,test_size])
train_data_loader=DataLoader(dataset=train_dataset,batch_size=32,shuffle=True)
test_data_loader=DataLoader(dataset=test_datset,batch_size=32,shuffle=True)

"""
model set and train
"""
model=CustomLinear(input_dim=1,hidden_dim=32,output_dim=1)
Model_Trainer.train(model=model,data_loader=train_data_loader,lr=0.0005,epochs=100)
Model_Trainer.test(model=model,data_loader=test_data_loader)