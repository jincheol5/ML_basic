import os
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

current_path=os.getcwd() # 현재 작업 경로
parent_path=os.path.dirname(current_path) # 현재 작업 경로의 부모 경로
grandparent_path=os.path.dirname(parent_path) # parent_path의 부모 경로
file_path = os.path.join(grandparent_path, "data") # data 파일 상대 경로 

# 참조 : https://eunhye-zz.tistory.com/8#google_vignette


class StockDataProcess:
    def __init__(self,seq_length):
        self.df=pd.read_csv(file_path+"/stock.csv")
        self.seq_length=seq_length

        print(self.df.shape,"\n")

        ### Data load
        # 주식 시장의 경우 최근 가격에서 과거로 거슬러 올라가며 패턴을 분석할 수 있다. -> 데이터 역순으로 정렬하여 최근 데이터 우선 학습
        # 데이터를 역순으로 정렬하여 전체 데이터의 70% 학습, 30% 테스트에 사용
        self.df = self.df[::-1] # df의 마지막 행이 첫 번재 행으로, 첫 번째 행이 마지막 행으로 가도록 변경
        train_size = int(len(self.df)*0.7)
        self.train_set = self.df[0:train_size]  
        self.test_set = self.df[train_size-self.seq_length:] # train_size-seq_length부터 시작 -> 시계열 데이터의 연속성을 유지하면서 학습 데이터와 테스트 데이터를 자연스럽게 연결


        # ### Data scaling
        # # input scale
        # scaler_x = MinMaxScaler()
        # scaler_x.fit(self.train_set.iloc[:, :-1]) # train_set의 모든 행과 마지막 열을 제외한 모든 열을 사용하여 스케일러의 최소값과 최대값 설정

        # self.train_set.iloc[:, :-1] = scaler_x.transform(self.train_set.iloc[:, :-1]) # 학습한 스케일러를 사용하여 train_set의 입력 데이터를 변환. 모든 행과 마지막 열을 제외한 모든 열을 스케일링
        # self.test_set.iloc[:, :-1] = scaler_x.transform(self.test_set.iloc[:, :-1]) # 동일한 스케일러를 사용하여 test_set의 입력 데이터를 변환. 이렇게 함으로써 테스트 데이터도 학습 데이터와 동일한 스케일로 변환

        # # Output scale
        # scaler_y = MinMaxScaler()
        # scaler_y.fit(self.train_set.iloc[:, [-1]]) # train_set의 마지막 열을 사용하여 스케일러를 학습

        # self.train_set.iloc[:, -1] = scaler_y.transform(self.train_set.iloc[:, [-1]]) # 출력 데이터 변환
        # self.test_set.iloc[:, -1] = scaler_y.transform(self.test_set.iloc[:, [-1]]) # 출력 데이터 변환

    def change_dataset_to_nparray(self):

        train_X=[]
        train_y=[]

        for index in range(0,len(self.train_set)-self.seq_length):
            _x=self.train_set.iloc[index:index+self.seq_length,:]
            _y=self.train_set.iloc[index+self.seq_length, [-1]]
            train_X.append(_x)
            train_y.append(_y)

        test_X=[]
        test_y=[]

        for index in range(0,len(self.test_set)-self.seq_length):
            _x=self.test_set.iloc[index:index+self.seq_length,:]
            _y=self.test_set.iloc[index+self.seq_length, [-1]]
            test_X.append(_x)
            test_y.append(_y)
        
        return np.array(train_X),np.array(train_y),np.array(test_X),np.array(test_y)

    def change_dataset_to_TensorDataset(self):
        train_X, train_y, test_X, test_y  = self.change_dataset_to_nparray()

        # 텐서로 변환
        train_X_tensor=torch.FloatTensor(train_X)
        train_y_tensor=torch.FloatTensor(train_y)

        test_X_tensor=torch.FloatTensor(test_X)
        test_y_tensor=torch.FloatTensor(test_y)

        # TensorDataset
        train_dataset=TensorDataset(train_X_tensor, train_y_tensor)
        test_dataset=TensorDataset(test_X_tensor, test_y_tensor)

        return train_dataset,test_dataset

    def get_DataLoader(self,batch_size):

        train_dataset,test_dataset=self.change_dataset_to_TensorDataset()

        train_DataLoader=DataLoader(train_dataset,batch_size=batch_size,shuffle=True,drop_last=True)

        test_DataLoader=DataLoader(test_dataset,batch_size=batch_size,shuffle=True,drop_last=True)

        return train_DataLoader,test_DataLoader