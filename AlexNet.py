import torch
import torch.nn as nn


class AlexNet(nn.Module):
    def __init__(self, n_class):
        super(AlexNet, self).__init__()
        self.n_class = n_class

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=(11, 11), stride=(4, 4))
        self.conv2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=(5, 5), padding=(2, 2))
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=(3, 3), padding=(1, 1))
        self.conv4 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=(3, 3), padding=(1, 1))
        self.conv5 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=(3, 3), padding=(1, 1))
        self.pool = nn.MaxPool2d(kernel_size=(3, 3), stride=2)

        self.fc1 = nn.Linear(in_features=5*5*256, out_features=4096)
        self.fc2 = nn.Linear(in_features=4096, out_features=4096)
        self.cls = nn.Linear(in_features=4096, out_features=self.n_class)

        self.dp = nn.Dropout(p=0.5)

    def forward(self, x):

        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))
        x = torch.relu(self.conv5(x))
        x = self.pool(x)
        x = self.dp(torch.relu(self.fc1(x.view(-1, 5*5*256))))
        x = self.dp(torch.relu(self.fc2(x)))
        x = self.cls(x)

        return x