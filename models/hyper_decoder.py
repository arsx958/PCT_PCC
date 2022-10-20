import torch.nn as nn
import torch
import torch.nn.functional as F


class HyperDecoder(nn.Module):
    def __init__(self, input_dim, outputdim=None):
        super(HyperDecoder, self).__init__()
        self.input_dim = input_dim
        self.fc1 = nn.Linear(input_dim, input_dim)
        self.fc2 = nn.Linear(input_dim, input_dim * 8)
        if not outputdim:
            self.fc3 = nn.Linear(input_dim * 8, input_dim * 32)
        else:
            self.fc3 = nn.Linear(input_dim * 8, outputdim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.exp(self.fc3(x))
        # x = F.relu(self.fc3(x))
        return x
