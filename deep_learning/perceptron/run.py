import torch
from perceptron import Perceptron

### simple dataset ###
# 입력과 타겟 정의
X = torch.tensor([[0.0,0.0],[0.0,1.0],[1.0,0.0],[1.0,1.0]])
y = torch.tensor([[-1],[1],[1],[1]])

# 가중치와 바이어스 정의
w = torch.tensor([[1.0], [1.0]], requires_grad=True) # gradient 계산 가능한 tensor
b = torch.tensor([-0.5], requires_grad=True) # gradient 계산 가능한 tensor

### train ###
model=Perceptron(input_dim=2, output_dim=1)
optimizer=torch.optim.Adam(model.parameters(),lr=0.1)

epoch=500
loss_fn=torch.nn.MSELoss() # 손실 함수 -> MSE 사용

for iter in range(epoch):
    pred=model(X) # 모델 순전파 -> 예측값 반환
    cost=loss_fn(pred,y) # 손실 계산
    optimizer.zero_grad() # 이전 epoch에서 계산된 gradient 초기화
    cost.backward() # 오류 역전파 
    optimizer.step() # 계산된 gradient를 사용하여 model의 parameter 업데이트

### result ###
pred=model(X)