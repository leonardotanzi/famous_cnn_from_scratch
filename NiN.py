import torch.nn as nn
import torch


class NiNblock(nn.Module):
    def __init__(self, in_features, out_features, kernel, stride, padding):
        super(NiNblock, self).__init__()
        self.conv1 = nn.Conv2d(in_features, out_features, kernel_size=kernel, stride=stride, padding=padding)
        self.conv1x1 = nn.Conv2d(out_features, out_features, (1, 1), stride=(1, 1), padding=0)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv1x1(x))
        x = torch.relu(self.conv1x1(x))
        return x


class NiN(nn.Module):
    def __init__(self, n_class):
        super(NiN, self).__init__()
        self.n_class = n_class
        self.block1 = NiNblock(3, 96, (11, 11), 4, 0)
        self.block2 = NiNblock(96, 256, (5, 5), 1, 2)
        self.block3 = NiNblock(256, 384, (3, 3), 1, 1)
        self.block4 = NiNblock(384, self.n_class, (3, 3), 1, 1)

        self.pool = nn.MaxPool2d((3, 3), stride=2)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fl = nn.Flatten(1)

    def forward(self, x):
        x = self.pool(self.block1(x))
        x = self.pool(self.block2(x))
        x = self.pool(self.block3(x))
        x = self.gap(self.block4(x))
        x = self.fl(x)
        return x