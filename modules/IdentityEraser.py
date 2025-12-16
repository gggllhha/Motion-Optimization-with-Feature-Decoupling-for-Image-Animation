from torch import nn
import torch.nn.functional as F


class IdentityEraser(nn.Module):
    def __init__(self, input_dim=512, hidden_dims=(1024, 1024), num_identities=246, **kwargs):
        super(IdentityEraser, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc3 = nn.Linear(hidden_dims[1], num_identities)
        self.norm1 = nn.LayerNorm(hidden_dims[0])
        self.norm2 = nn.LayerNorm(hidden_dims[1])
        self.act = nn.LeakyReLU(0.2, inplace=False)
        for m in (self.fc1, self.fc2, self.fc3):
            nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')

    def forward(self, w_m):
        x = self.act(self.norm1(self.fc1(w_m)))
        x = self.act(self.norm2(self.fc2(x)))
        pred = self.fc3(x)  # b, C

        return pred
