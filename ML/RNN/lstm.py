import torch
import torch.nn as nn


class CustomLSTMCell(nn.Module):
    def __init__(self,input_dim,hidden_dim):
        super().__init__()
        self.input_dim=input_dim
        self.hidden_dim=hidden_dim

        # forget gate
        self.W_hf=nn.Linear(self.hidden_dim,self.hidden_dim)
        self.W_xf=nn.Linear(self.input_dim,self.hidden_dim)

        # input gate
        self.W_hi=nn.Linear(self.hidden_dim,self.hidden_dim)
        self.W_xi=nn.Linear(self.input_dim,self.hidden_dim)

        # candidate cell state gate 
        self.W_hg=nn.Linear(self.hidden_dim,self.hidden_dim)
        self.W_xg=nn.Linear(self.input_dim,self.hidden_dim)

        # output gate
        self.W_ho=nn.Linear(self.hidden_dim,self.hidden_dim)
        self.W_xo=nn.Linear(self.input_dim,self.hidden_dim)

        self.act_tanh=nn.Tanh()
        self.act_sigmoid=nn.Sigmoid()

    def forward(self,x_seq):
        # input x_seq = [batch_size,seq_len,input_dim]

        batch_size,seq_len,input_dim=x_seq.size()

        device=x_seq.device # Ensure all tensors are on the correct device

        h_0=torch.zeros(self.hidden_dim,device=device)

        c_0=torch.zeros(self.hidden_dim,device=device)

        output_h_t=torch.zeros(batch_size,self.hidden_dim,device=device)

        for batch in range(batch_size):

            prev_h_t=h_0
            prev_c_t=c_0

            h_t=h_0
            c_t=c_0

            for t in range(seq_len):
                
                # forget gate
                f_t=self.act_sigmoid(self.W_hf(prev_h_t)+self.W_xf(x_seq[batch][t]))
                
                # input gate
                i_t=self.act_sigmoid(self.W_hi(prev_h_t)+self.W_xi(x_seq[batch][t]))
                
                # candidate cell state
                g_t=self.act_tanh(self.W_hg(prev_h_t)+self.W_xg(x_seq[batch][t]))

                # output gate
                o_t=self.act_sigmoid(self.W_ho(prev_h_t)+self.W_xo(x_seq[batch][t]))

                # cell state
                c_t=f_t*prev_c_t+i_t*g_t # 요소별 합과 요소별 곱 
                
                # output
                h_t=o_t*self.act_tanh(c_t)

                prev_h_t=h_t
                prev_c_t=c_t

            output_h_t[batch]=h_t

        return output_h_t


# many-to-one rnn model
class CustomLSTM(nn.Module):
    def __init__(self,input_dim,hidden_dim,output_dim):
        super().__init__()
        self.input_dim=input_dim
        self.hidden_dim=hidden_dim
        self.output_dim=output_dim

        self.lstm_cell=CustomLSTMCell(self.input_dim,self.hidden_dim)
        self.fc=nn.Linear(self.hidden_dim,self.output_dim)

    def forward(self,x_seq):
        # input x_seq = [batch_size,seq_len,input_dim] 형태이다. 
        hidden_state=self.lstm_cell(x_seq)
        return self.fc(hidden_state)

# many-to-one rnn model
class PytorchLSTM(nn.Module):
    def __init__(self,input_dim,hidden_dim,output_dim):
        super().__init__()
        self.input_dim=input_dim
        self.hidden_dim=hidden_dim
        self.output_dim=output_dim
        self.num_layers=1

        self.lstm=nn.LSTM(self.input_dim,self.hidden_dim,num_layers=self.num_layers,batch_first=True)
        self.fc=nn.Linear(self.hidden_dim,self.output_dim)

    def forward(self,x_seq):
        # input x_seq = [batch_size,seq_len,input_dim] 형태이다.
        # nn.LSTM의 입력 형태는 [seq_len,batch_size,dim] 형태이다.
        # 따라서, batch_first 옵션을 사용하여 입력 형태가 [batch_size,seq_len,input_dim] 로 바뀐다.
        
        batch_size,seq_len,input_dim=x_seq.size()

        device = x_seq.device  # Ensure all tensors are on the correct device
        h_0=torch.zeros(self.num_layers,batch_size,self.hidden_dim,device=device)
        c_0=torch.zeros(self.num_layers,batch_size,self.hidden_dim,device=device)

        # nn.LSTM의 반환값 = output (all time step에 대한 출력), hn (마지막 time step에서 각 층과 각 방향의 hidden state), ct (cell state)
        # batch_first=true인 경우 output=[batch_size,seq_len,output_dim]
        output,(hn, cn) = self.lstm(x_seq, (h_0,c_0))

        return self.fc(output[:, -1, :])