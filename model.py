import torch
torch.manual_seed(1234)


class SingleMLP_Classifier(torch.nn.Module):
    def __init__(self, input_shape, dropout = 0.5):
        super().__init__()
        self.dropout = dropout
        
        self.linear_relu_stack = torch.nn.Sequential(
            torch.nn.Linear(input_shape, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(self.dropout),
            torch.nn.Linear(256, 2)
            )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits


class ResidualBlock(torch.nn.Module):
    def __init__(self, hidden_dim):
        super(ResidualBlock, self).__init__()
        self.fc1 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.relu = torch.nn.ReLU()
        # self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        identity = x
        
        out = self.fc1(x)
        # out = self.relu(out)
        # out = self.fc2(out)
        
        out = out + identity
        out = self.relu(out)
        
        return out

class DNN_Classifier(torch.nn.Module):
    def __init__(self, input_size, hidden_size=256):
        super(DNN_Classifier, self).__init__()
        
        self.layer1 = torch.nn.Linear(input_size, hidden_size)
        self.relu = torch.nn.ReLU()

        self.res_block = ResidualBlock(hidden_size)

        self.layer3 = torch.nn.Linear(hidden_size, 2)
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        
        x = self.res_block(x)

        x = self.layer3(x)
        
        return x