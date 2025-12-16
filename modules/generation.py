import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """一个两层 MLP，用来从 F (512‑d) 计算出某一层的 style 向量。"""

    def __init__(self, input_dim=512, hidden_dim=512, output_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)  # (bs, output_dim)


class ModulatedConv2d(nn.Module):
    """
    StyleGAN2-style modulated convolution:
      - Weight modulation from style
      - Weight demodulation for normalization
      - Static learnable bias per output channel
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, style_dim=512, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.weight = nn.Parameter(
            torch.randn(1, out_channels, in_channels, kernel_size, kernel_size)
        )
        self.modulation = nn.Linear(style_dim, in_channels)
        self.bias = nn.Parameter(torch.zeros(out_channels))

        self.padding = kernel_size // 2

    def forward(self, x, w):
        batch, _, height, width = x.shape

        style_scale = self.modulation(w).view(batch, 1, self.in_channels, 1, 1)
        style_scale = style_scale + 1.0

        weight = self.weight * style_scale

        demod = torch.rsqrt((weight * weight).sum(dim=[2, 3, 4]) + self.eps)
        weight = weight * demod.view(batch, self.out_channels, 1, 1, 1)

        x = x.view(1, batch * self.in_channels, height, width)
        weight = weight.view(batch * self.out_channels, self.in_channels, self.kernel_size, self.kernel_size)
        out = F.conv2d(x, weight, padding=self.padding, groups=batch)
        out = out.view(batch, self.out_channels, height, width)

        out = out + self.bias.view(1, -1, 1, 1)
        return out


class GenerationModule(nn.Module):
    def __init__(self, style_dim=512, fmap_base=4096, num_mlps=6, max_features=512, **kwargs):
        """
        :param style_dim: F, w_id, w_m 的维度 (512)
        :param fmap_base: 网络最初的 feature map 通道数
        :param num_mlps: MLP 数量，也对应 ModulatedConv2d 层数
        """
        super().__init__()

        self.mlps = nn.ModuleList([
            MLP(input_dim=style_dim, hidden_dim=style_dim, output_dim=style_dim)
            for _ in range(num_mlps)
        ])

        self.mods = nn.ModuleList([
            ModulatedConv2d(
                in_channels=min(fmap_base // (2 ** i), max_features),
                out_channels=min(fmap_base // (2 ** (i + 1)), max_features),
                kernel_size=3,
                style_dim=style_dim
            )
            for i in range(num_mlps)
        ])
        self.const = nn.Parameter(torch.randn(1, 512, 4, 4))
        self.to_rgb = nn.Conv2d(fmap_base // (2 ** num_mlps), 3, kernel_size=(7, 7), padding=(3, 3))

    def forward(self, w_id, w_m):
        f = w_id + w_m  # (bs, 512)
        bs = f.size(0)
        x = self.const.repeat(bs, 1, 1, 1)  # (bs, C, 4, 4)
        for mlp, mod in zip(self.mlps, self.mods):
            style = mlp(f)  # (bs, 512)
            x = F.interpolate(x, scale_factor=2, mode='nearest')
            x = mod(x, style)
            x = F.leaky_relu(x, 0.2, inplace=True)

        img = self.to_rgb(x)  # (bs, 3, H, W)
        img = torch.sigmoid(img)
        return img
