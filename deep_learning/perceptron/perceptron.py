import torch 

class Perceptron(torch.nn.Module):
    def __init__(self,input_dim,output_dim):
        super(Perceptron, self).__init__()

        self.perceptron = torch.nn.Linear(in_features=input_dim,out_features=output_dim,bias=True) # perceptron layer

        self.activation = torch.nn.ReLU() # activation function ReLU

        self.model=torch.nn.Sequential(self.perceptron,self.activation)

    def forward(self, x):
        return self.model(x)
