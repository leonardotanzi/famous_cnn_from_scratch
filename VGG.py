import torch.nn as nn
import torch


class VGGblock(nn.Module):
    def __init__(self, num_convs, in_features, out_features):
        super(VGGblock, self).__init__()
        self.num_convs = num_convs
        self.conv = nn.Conv2d(in_features, out_features, kernel_size=(3, 3), padding=1)
        self.conv_same = nn.Conv2d(out_features, out_features, kernel_size=(3, 3), padding=1)
        self.pool = nn.MaxPool2d((2, 2), stride=2)

    def forward(self, x):
        for i in range(self.num_convs):
            x = torch.relu(self.conv(x)) if i == 0 else torch.relu(self.conv_same(x))
        x = self.pool(x)
        return x


class VGG(nn.Module):
    def __init__(self, n_class):
        super(VGG, self).__init__()
        self.n_class = n_class
        self.block1 = VGGblock(1, 3, 64)
        self.block2 = VGGblock(1, 64, 128)
        self.block3 = VGGblock(2, 128, 256)
        self.block4 = VGGblock(2, 256, 512)
        self.block5 = VGGblock(2, 512, 512)

        self.fc1 = nn.Linear(512*7*7, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.cls = nn.Linear(4096, self.n_class)

        self.dp = nn.Dropout(0.5)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.dp(torch.relu(self.fc1(x.view(-1, 512*7*7))))
        x = self.dp(torch.relu(self.fc2(x)))
        x = self.cls(x)
        return x
