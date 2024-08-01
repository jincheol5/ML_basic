from rnn import CustomRNN,PytorchRNN
from custom_dataset import StockDataProcess
from model_optimization import Optimization

import os
import numpy as np
import random
import torch
import torch.nn as nn



# random seed
random_seed= 42
random.seed(random_seed)
np.random.seed(random_seed)
os.environ["PYTHONHASHSEED"] = str(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(False)

# GPU 사용 가능한지 확인
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("GPU is available\n")
else:
    device = torch.device("cpu")
    print("GPU is not available, using CPU instead\n")

# hyperparameter
lr=0.01
batch_size=100
epochs=100

### custom RNN
# model
custom_model=CustomRNN(input_dim=5,hidden_dim=10,output_dim=1).to(device)

# get DataLoader
SDP=StockDataProcess(seq_length=7)
train_DataLoader,test_DataLoader=SDP.get_DataLoader(batch_size=batch_size) 

# optimize model
optimization=Optimization(model=custom_model,lr=lr)
for epoch in range(epochs):
    # train model
    optimization.train_model(dataloader=train_DataLoader,loss_fn=nn.MSELoss())

    # test model
    optimization.test_model(dataloader=test_DataLoader,loss_fn=nn.MSELoss())

### pytorch RNN
# model
pytorch_model=PytorchRNN(input_dim=5,hidden_dim=10,output_dim=1).to(device)

# get DataLoader
SDP=StockDataProcess(seq_length=7)
train_DataLoader,test_DataLoader=SDP.get_DataLoader(batch_size=batch_size) 

# optimize model
optimization=Optimization(model=pytorch_model,lr=lr)
for epoch in range(epochs):
    # train model
    optimization.train_model(dataloader=train_DataLoader,loss_fn=nn.MSELoss())

    # test model
    optimization.test_model(dataloader=test_DataLoader,loss_fn=nn.MSELoss())

