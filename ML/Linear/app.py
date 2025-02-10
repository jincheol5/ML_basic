import os
import random
import numpy as np
import yaml
import wandb
from model import CustomLinear
from dataset import CustomDataset
from model_train import Model_Trainer
import torch
from torch.utils.data import random_split,DataLoader

"""
1. get sweep_config dict from yaml file
2. get sweep_id using wandb.sweep()
"""
with open('./sweep_config.yaml') as file:
    sweep_config=yaml.safe_load(file)
sweep_id=wandb.sweep(sweep=sweep_config,project='ML_basic')

def main():
    """
    wandb.init()
    """
    wandb.init()
    """
    seed setting
    """
    random.seed(wandb.config.seed)
    np.random.seed(wandb.config.seed)
    torch.manual_seed(wandb.config.seed) 
    os.environ["PYTHONHASHSEED"]=str(wandb.config.seed)
    torch.cuda.manual_seed(wandb.config.seed)
    torch.cuda.manual_seed_all(wandb.config.seed)
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
    train_data_loader=DataLoader(dataset=train_dataset,batch_size=wandb.config.batch_size,shuffle=True)
    test_data_loader=DataLoader(dataset=test_dataset,batch_size=wandb.config.batch_size,shuffle=True)

    """
    model set and train
    """
    model=CustomLinear(input_dim=1,output_dim=1)
    Model_Trainer.train(model=model,data_loader=train_data_loader,config=wandb.config)
    Model_Trainer.test(model=model,data_loader=test_data_loader)
"""
execute wandb.agent() 
"""
wandb.agent(sweep_id=sweep_id,function=main,count=10)