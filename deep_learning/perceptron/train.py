import torch
from perceptron import Perceptron

def train_epoch(dataloader,optimizer,model,loss_fn):
    model.train() # model을 학습 모드로 변경
    for X,y in dataloader:

        pred=model(X) # 모델 순전파 -> 예측값 반환
        cost=loss_fn(pred,y) # 손실 계산
        optimizer.zero_grad() # 이전 epoch에서 계산된 gradient 초기화
        cost.backward() # 오류 역전파 
        optimizer.step() # 계산된 gradient를 사용하여 model의 parameter 업데이트
