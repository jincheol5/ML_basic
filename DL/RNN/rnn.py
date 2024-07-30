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
        # input x_seq = [batch_size,seq_len,input_dim]

        batch_size,seq_len,input_dim=x_seq.size()

        device=x_seq.device # Ensure all tensors are on the correct device

        h_0=torch.zeros(self.hidden_dim,device=device)

        output_h_t=torch.zeros(batch_size,self.hidden_dim,device=device)

        print(f"h_0 is on {h_0.device}, expected {device}")
        print(f"output_h_t is on {output_h_t.device}, expected {device}")

        for batch in range(batch_size):
            prev_h_t=h_0
            h_t=h_0

            print(f"prev_h_t is on {prev_h_t.device}, expected {device}")
            print(f"h_t is on {h_t.device}, expected {device}")

            print(f"x_seq[batch][t] is on {x_seq[batch][t].device}, expected {device}")

            for t in range(seq_len):
                h_t=self.act(self.W_h(prev_h_t)+self.W_x(x_seq[batch][t]))
                prev_h_t=h_t

            output_h_t[batch]=h_t

        return output_h_t


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
        # input x_seq = [batch_size,seq_len,input_dim] 형태이다. 
        hidden_state=self.rnn_cell(x_seq)
        return self.fc(hidden_state)
    

# many-to-one rnn model
class PytorchRNN(nn.Module):
    def __init__(self,input_dim,hidden_dim,output_dim):
        super().__init__()
        self.input_dim=input_dim
        self.hidden_dim=hidden_dim
        self.output_dim=output_dim

        self.rnn_cell=nn.RNN(self.input_dim,self.hidden_dim,batch_first=True)
        self.fc=nn.Linear(self.hidden_dim,self.output_dim)

    def forward(self,x_seq):
        # input x_seq = [batch_size,seq_len,input_dim] 형태이다.
        # nn.RNN의 입력 형태는 [seq_len,batch_size,dim] 형태이다.
        # 따라서, batch_first 옵션을 사용하여 입력 형태가 [batch_size,seq_len,input_dim] 로 바뀐다.
        
        device = x_seq.device  # Ensure all tensors are on the correct device
        h_0=torch.zeros(self.hidden_dim,device=device)

        # nn.RNN의 반환값 = output (all time step에 대한 출력), hn (마지막 time step에서 각 층과 각 방향의 hidden state)
        # batch_first=true인 경우 output=[batch_size,seq_len,output_dim]
        output,_ = self.rnn(x_seq, h_0)

        return self.fc(output)
