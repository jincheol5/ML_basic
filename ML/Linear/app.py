import os
import random
import numpy as np
import wandb
from model import CustomLinear
from dataset import CustomDataset
from model_train import Model_Trainer
import torch
from torch.utils.data import random_split,DataLoader
"""
wandb
"""
wandb.init(project='ML_basic',config={
    'seed':42,
    'lr':0.0005,
    'epochs':1000,
    'batch_size':32
})
config=wandb.config

"""
seed setting
"""
random.seed(config['seed'])
np.random.seed(config['seed'])
torch.manual_seed(config['seed']) 
os.environ["PYTHONHASHSEED"]=str(config['seed'])
torch.cuda.manual_seed(config['seed'])
torch.cuda.manual_seed_all(config['seed'])
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
train_dataset,test_dataset=random_split(dataset=dataset,lengths=[train_size,test_size])
train_data_loader=DataLoader(dataset=train_dataset,batch_size=config['batch_size'],shuffle=True)
test_data_loader=DataLoader(dataset=test_dataset,batch_size=config['batch_size'],shuffle=True)

"""
model set and train
"""
model=CustomLinear(input_dim=1,output_dim=1)
Model_Trainer.train(model=model,data_loader=train_data_loader,config=config)
Model_Trainer.test(model=model,data_loader=test_data_loader)