import torch.nn as nn
import torch
import torch.nn.functional as F


class HyperEncoder(nn.Module):
    def __init__(self, input_dim):
        super(HyperEncoder, self).__init__()
        self.input_dim = input_dim
        self.fc1 = nn.Linear(input_dim, input_dim)
        self.fc2 = nn.Linear(input_dim, input_dim // 4)
        self.fc3 = nn.Linear(input_dim // 4, input_dim // 32)

    def forward(self, x):
        x = torch.abs(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
