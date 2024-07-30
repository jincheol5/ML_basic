import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# GPU 사용 가능한지 확인
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

class Optimization:

    def __init__(self,model,epochs,lr):
        self.model=model
        self.epochs=epochs
        self.lr=lr
        self.optimizer=optim.Adam(self.model.parameters(), lr = self.lr)

    def train_model(self,dataloader,loss_fn):
        
        self.model.train()

        for batch_idx,(inputs,targets) in enumerate(tqdm(dataloader)):
            
            inputs.to(device)
            targets.to(device)

            print(inputs.device)
            print(targets.device)

            # 예측과 손실 계산
            pred=self.model(inputs)
            loss=loss_fn(pred,targets)

            # 역전파
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
        
        print("model train end\n")
    
    def test_model(self,dataloader,loss_fn):

        self.model.eval()

        # torch.no_grad()를 사용하여 테스트 시 변화도(gradient)를 계산하지 않도록 합니다.
        # 이는 requires_grad=True로 설정된 텐서들의 불필요한 변화도 연산 및 메모리 사용량 또한 줄여줍니다.
        with torch.no_grad():
            for batch_idx,(inputs,targets) in enumerate(tqdm(dataloader)):

                inputs.to(device)
                targets.to(device)

                pred = self.model(inputs)
                mse_loss = loss_fn(pred,targets)
                print((batch_idx+1)," batch Mean Squared Error: ",mse_loss,"\n")
                
        print("model test end\n")


