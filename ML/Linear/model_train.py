from tqdm import tqdm
import numpy as np
import torch
from torch.nn import MSELoss

class Model_Trainer:
    @staticmethod
    def train(model,data_loader,lr=0.0005,epochs=10):
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        optimizer=torch.optim.Adam(model.parameters(),lr=lr)
        model.to(device)
        model.train()
        for epoch in tqdm(range(epochs),desc=f"model training..."):
            for batch in data_loader:
                inputs,labels=batch
                inputs=inputs.to(device)
                labels=labels.to(device)
                outputs=model(x=inputs)
                loss=MSELoss(outputs,labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    @staticmethod
    def test(model,data_loader):
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.eval()
        with torch.no_grad():
            mse_loss_list=[]
            for batch in tqdm(data_loader,desc=f"model evaluating..."):
                inputs,labels=batch
                inputs=inputs.to(device)
                labels=labels.to(device)
                outputs=model(x=inputs)
                mse_loss=MSELoss(outputs,labels)
                mse_loss_list.append(mse_loss.item())
        print(f"MSE Loss: {np.mean(mse_loss_list)}")