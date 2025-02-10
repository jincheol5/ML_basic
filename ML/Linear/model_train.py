from tqdm import tqdm
import numpy as np
import wandb
import torch
from torch.nn import MSELoss

class Model_Trainer:
    @staticmethod
    def train(model,data_loader,config):
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        optimizer=torch.optim.Adam(model.parameters(),lr=config.lr) if config.optimizer=='adam' else torch.optim.SGD(model.parameters(),lr=config.lr)
        model.to(device)
        model.train()
        criterion=MSELoss()
        for epoch in tqdm(range(config.epochs),desc=f"model training..."):
            sum_batch_loss=0.0
            for batch in data_loader:
                inputs,labels=batch
                inputs=inputs.to(device)
                labels=labels.to(device)
                outputs=model(x=inputs)
                loss=criterion(outputs,labels)
                sum_batch_loss+=loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            epoch_loss=sum_batch_loss/len(data_loader)
            wandb.log({"mse_loss": epoch_loss},step=epoch)
        wandb.finish()

    @staticmethod
    def test(model,data_loader):
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.eval()
        criterion=MSELoss()
        total_loss=[]
        with torch.no_grad():
            for batch in tqdm(data_loader,desc=f"model evaluating..."):
                inputs,labels=batch
                inputs=inputs.to(device)
                labels=labels.to(device)
                outputs=model(x=inputs)
                loss=criterion(outputs,labels)
                total_loss.append(loss.item())
        print(f"MSE Loss: {np.mean(total_loss)}")