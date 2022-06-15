import torch
import torch.nn as nn


class InceptionBlock(nn.Module):
    def __init__(self, in_channels, c1, c2, c3, c4):
        super(InceptionBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, c1, kernel_size=(1, 1), stride=(1, 1))
        self.conv3_reduce = nn.Conv2d(in_channels, c2[0], kernel_size=(1, 1))
        self.conv3 = nn.Conv2d(c2[0], c2[1], kernel_size=(3, 3), padding=1)
        self.conv5_reduce = nn.Conv2d(in_channels, c3[0], kernel_size=(1, 1))
        self.conv5 = nn.Conv2d(c3[0], c3[1], kernel_size=(5, 5), padding=2)

        self.maxpool = nn.MaxPool2d((3, 3), stride=(1, 1), padding=1)
        self.maxpool_reduce = nn.Conv2d(in_channels, c4, kernel_size=(1, 1))

    def forward(self, x):
        x1 = torch.relu(self.conv1(x))

        x2 = torch.relu(self.conv3_reduce(x))
        x2 = torch.relu(self.conv3(x2))

        x3 = torch.relu(self.conv5_reduce(x))
        x3 = torch.relu(self.conv5(x3))

        x4 = self.maxpool(x)
        x4 = torch.relu(self.maxpool_reduce(x4))

        return torch.cat((x1, x2, x3, x4), dim=1)


class Inception(nn.Module):
    def __init__(self, n_classes):
        super(Inception, self).__init__()
        self.n_classes = n_classes
        self.conv1 = nn.Conv2d(3, 64, (7, 7), stride=(2, 2), padding=3)
        self.maxpool = nn.MaxPool2d((3, 3), stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 64, (1, 1))
        self.conv3 = nn.Conv2d(64, 192, (3, 3), padding=1)
        self.block1_1 = InceptionBlock(192, 64, (96, 128), (16, 32), 32)
        self.block1_2 = InceptionBlock(256, 128, (128, 192), (32, 96), 64)

        self.block2_1 = InceptionBlock(480, 192, (96, 208), (16, 48), 64)
        self.block2_2 = InceptionBlock(512, 160, (112, 224), (24, 64), 64)
        self.block2_3 = InceptionBlock(512, 128, (128, 256), (24, 64), 64)
        self.block2_4 = InceptionBlock(512, 112, (144, 288), (32, 64), 64)
        self.block2_5 = InceptionBlock(528, 256, (160, 320), (32, 128), 128)

        self.block3_1 = InceptionBlock(832, 256, (160, 320), (32, 128), 128)
        self.block3_2 = InceptionBlock(832, 384, (192, 384), (48, 128), 128)

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1024, self.n_classes)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.maxpool(x)
        x = torch.relu(self.conv3(x))
        x = self.maxpool(x)
        x = self.block1_1(x)
        x = self.block1_2(x)
        x = self.maxpool(x)
        x = self.block2_1(x)
        x = self.block2_2(x)
        x = self.block2_3(x)
        x = self.block2_4(x)
        x = self.block2_5(x)
        x = self.maxpool(x)
        x = self.block3_1(x)
        x = self.block3_2(x)
        x = self.gap(x)
        x = self.fc(x.view(-1, 1024))
        return x