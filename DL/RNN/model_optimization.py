import torch
import torch.optim as optim
from tqdm import tqdm


class Optimization:

    def __init__(self,model,epochs,lr):
        self.model=model
        self.epochs=epochs
        self.lr=lr
        self.optimizer=optim.Adam(self.model.parameters(), lr = self.lr)
        

    def train_model(self,dataloader,loss_fn):
        
        self.model.train()

        for batch_idx,(inputs,targets) in tqdm(enumerate(dataloader)):
            
            print(inputs)
            print(targets)

            # 예측과 손실 계산
            pred=self.model(inputs)
            loss=loss_fn(pred,targets)

            # 역전파
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
    
    def test_model(self,dataloader,loss_fn):

        self.model.test()
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        test_loss, correct = 0, 0

        # torch.no_grad()를 사용하여 테스트 시 변화도(gradient)를 계산하지 않도록 합니다.
        # 이는 requires_grad=True로 설정된 텐서들의 불필요한 변화도 연산 및 메모리 사용량 또한 줄여줍니다.
        with torch.no_grad():
            for inputs,targets in tqdm(dataloader):
                pred = self.model(inputs)
                test_loss += loss_fn(pred, targets).item()
                correct += (pred.argmax(1) == targets).type(torch.float).sum().item()

        test_loss /= num_batches
        correct /= size
        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
