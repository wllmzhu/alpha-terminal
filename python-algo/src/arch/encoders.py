import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialEncoder(nn.Module):
    def __init__(self): 
        super().__init__()
        self.conv1 = nn.Conv2d(7, 16, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.fc1 = nn.Linear(32 * 5 * 5, 512)
        self.fc2 = nn.Linear(512, 224)

    def forward(self, x):
        # (N, 7, 28, 28)
        # (N, 16, 13, 13)
        x = self.pool(F.relu(self.conv1(x)))
        # (N, 32, 5, 5)
        x = self.pool(F.relu(self.conv2(x)))
        # (N, 800)
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class ScalarEncoder(nn.Module):
    def __init__(self, scalar_features_dim=7):
        super().__init__()
        self.fc1 = nn.Linear(scalar_features_dim, 64)
        self.fc2 = nn.Linear(64, 32)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x