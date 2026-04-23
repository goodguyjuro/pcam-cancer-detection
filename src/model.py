import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    def __init__(self, use_batchnorm=False, use_dropout=False):
        super(SimpleCNN, self).__init__()
        self.use_batchnorm = use_batchnorm
        self.use_dropout = use_dropout

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16) if use_batchnorm else nn.Identity()
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32) if use_batchnorm else nn.Identity()
        self.pool = nn.MaxPool2d(2, 2)

        self.dropout = nn.Dropout(0.5) if use_dropout else nn.Identity()
        self.fc1 = nn.Linear(32 * 24 * 24, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def get_model(version: int = 1):
    if version == 1:
        return SimpleCNN(use_batchnorm=False, use_dropout=False)
    if version == 2:
        return SimpleCNN(use_batchnorm=False, use_dropout=False)
    if version == 3:
        return SimpleCNN(use_batchnorm=True, use_dropout=False)
    if version == 4:
        return SimpleCNN(use_batchnorm=True, use_dropout=True)

    raise ValueError(f"Unsupported model version: {version}")
