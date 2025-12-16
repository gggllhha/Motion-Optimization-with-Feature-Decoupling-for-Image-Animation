import torch
from torch import nn
import torch.nn.functional as F
from modules.util import DownBlock2d, SameBlock2d


class MLP_ID(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        """
        :param input_dim: 输入特征的维度 (来自 encoder)
        :param hidden_dims: 一个列表，包含每层的隐藏单元个数，例如 [256, 128]
        :param output_dim: 输出特征 a_id 的维度
        """
        super(MLP_ID, self).__init__()
        layers = []
        in_dim = input_dim
        for hidden_dim in hidden_dims:
            layer = nn.Linear(in_dim, hidden_dim)
            nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='leaky_relu')
            layers.append(layer)
            layers.append(nn.BatchNorm1d(hidden_dim, affine=True))
            layers.append(nn.LeakyReLU(0.2, inplace=False))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):

        x = self.pool(x)  # 输出形状 (bs, 512, 1, 1)
        x = x.view(x.size(0), -1)  # 展平为 (bs, 512)
        return self.net(x)


class MLP_Motion(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super(MLP_Motion, self).__init__()
        layers = []
        in_dim = input_dim
        for hidden_dim in hidden_dims:
            layer = nn.Linear(in_dim, hidden_dim)
            nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='leaky_relu')
            layers.append(layer)
            layers.append(nn.BatchNorm1d(hidden_dim, affine=True))
            layers.append(nn.LeakyReLU(0.2, inplace=False))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.pool(x)  # 输出形状 (bs, 512, 1, 1)
        x = x.view(x.size(0), -1)  # 展平为 (bs, 512)
        return self.net(x)


class Encoder(nn.Module):
    def __init__(self, block_expansion=64, in_channel=3, num_blocks=3, max_features=256):
        super(Encoder, self).__init__()
        self.first = SameBlock2d(in_channel, block_expansion, kernel_size=(7, 7), padding=(3, 3))
        down_blocks = []
        for i in range(num_blocks):
            in_features = min(max_features, block_expansion * (2 ** i))
            out_features = min(max_features, block_expansion * (2 ** (i + 1)))
            down_blocks.append(DownBlock2d(in_features, out_features, kernel_size=(3, 3), padding=(1, 1)))
        self.down_blocks = nn.ModuleList(down_blocks)

    def forward(self, x):
        x = self.first(x)
        for down_block in self.down_blocks:
            x = down_block(x)

        return x


class CoefficientExtractor(nn.Module):
    def __init__(self, block_expansion, in_channel, num_blocks, max_features):
        super().__init__()
        self.encoder = Encoder(block_expansion, in_channel, num_blocks, max_features)
        self.mlp_id = MLP_ID(input_dim=1024, hidden_dims=[512, 256], output_dim=20)
        self.mlp_motion = MLP_Motion(input_dim=1024, hidden_dims=[512, 256], output_dim=20)

    def forward(self, I_s, I_d):
        feat_s = self.encoder(I_s)
        feat_d = self.encoder(I_d)

        a_id = self.mlp_id(feat_s)
        b_m = self.mlp_motion(feat_d)

        return a_id, b_m


def gram_schmidt(matrix):
    q, dim = matrix.shape
    Q, R = torch.linalg.qr(matrix.T, mode='reduced')
    Q = Q.contiguous()
    return Q.T[:q]


class Disentanglement(nn.Module):
    def __init__(self, block_expansion=64, in_channel=3, num_blocks=6, max_features=1024, num_basis=40, dim=512,
                 **kwargs):
        super().__init__()
        self.q = num_basis // 2
        self.p = num_basis // 2
        self.dim = dim
        self.num_basis = num_basis
        self.D = nn.Parameter(torch.randn(self.num_basis, dim))
        self.CoefficientExtractor = CoefficientExtractor(block_expansion, in_channel, num_blocks, max_features)

    def forward(self, I_s, I_d):
        D_ortho = gram_schmidt(self.D)
        D1 = D_ortho[:20]
        D2 = D_ortho[20:]
        a_id, b_m = self.CoefficientExtractor(I_s, I_d)
        w_id = torch.einsum('...i,id->...d', a_id, D1)
        w_m = torch.einsum('...i,id->...d', b_m, D2)

        return w_id, w_m


