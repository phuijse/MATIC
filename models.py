import torch.nn as nn

class SimpleConv(nn.Module):

    def __init__(self, n_channels=1, n_classes=7):
        super(type(self), self).__init__()
        self.activation = nn.ReLU()
        self.conv1 = nn.Conv1d(n_channels, 32, 3, 2)
        self.conv2 = nn.Conv1d(32, 32, 3, 2)
        self.global_time_pooling = nn.AdaptiveMaxPool1d(1)
        self.linear1 = nn.Linear(32, 32)    
        self.linear2 = nn.Linear(32, n_classes)    

    def forward(self, x):
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.global_time_pooling(x)
        x = self.activation(self.linear1(x.view(-1, 32)))
        return self.linear2(x)
