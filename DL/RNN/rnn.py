import torch
import torch.nn as nn



class CustomRNNCell(nn.Module):
    def __init__(self,input_dim,hidden_dim):
        super().__init__()
        self.input_dim=input_dim
        self.hidden_dim=hidden_dim

        self.W_h=nn.Linear(self.hidden_dim,self.hidden_dim) # 이전 은닉 상태값에 대한 가중치 행렬 계산 층 생성
        self.W_x=nn.Linear(self.input_dim,self.hidden_dim) # 입력값에 대한 가중치 행렬 계산 층 생성

        self.act=nn.Tanh()

    def forward(self,x_seq):
        # x = input sequence (seq_len * input_dim)

        seq_len,_=x_seq.size()

        h_0=torch.zeros(self.hidden_dim)

        prev_h_t=h_0
        h_t=h_0

        for t in range(seq_len):
            h_t=self.act(self.W_h(prev_h_t)+self.W_x(x_seq[t]))
            prev_h_t=h_t

        return h_t


# many-to-one rnn model
class CustomRNN(nn.Module):
    def __init__(self,input_dim,hidden_dim,output_dim):
        super().__init__()
        self.input_dim=input_dim
        self.hidden_dim=hidden_dim
        self.output_dim=output_dim

        self.rnn_cell=CustomRNNCell(self.input_dim,self.hidden_dim)
        self.fc=nn.Linear(self.hidden_dim,self.output_dim)

    def forward(self,x_seq):
        hidden_state=self.rnn_cell(x_seq)
        return self.fc(hidden_state)