from torch import nn
import torch.nn.functional as F


class IdentityClassifier(nn.Module):
    def __init__(self, input_dim=512, hidden_dims=(1024, 1024), num_identities=246, **kwargs):
        super(IdentityClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc3 = nn.Linear(hidden_dims[1], num_identities)
        self.norm1 = nn.BatchNorm1d(hidden_dims[0], affine=True)
        self.norm2 = nn.BatchNorm1d(hidden_dims[1], affine=True)
        nn.init.kaiming_normal_(self.fc1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc2.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc3.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, w_m):
        x = F.relu(self.norm1(self.fc1(w_m)))
        x = F.relu(self.norm2(self.fc2(x)))
        pred = self.fc3(x)  # b, C

        return pred
