import torch
from torch import nn
import torch.nn.functional as F


class Residule(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.convs(x) + x


class FFNNetwork(nn.Module):
    def __init__(self, grid_size: int):
        super().__init__()

        n_classes = grid_size ** 2

        self.layers = nn.Sequential(
            Residule(n_classes),
            nn.Conv2d(n_classes, 128, kernel_size=3, padding=1),
            Residule(128),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            Residule(256),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            Residule(128),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            Residule(64),
            nn.Conv2d(64, 1, kernel_size=3, padding=1),
            nn.Flatten(),
            nn.Linear(grid_size**2, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 4),
        )

    def forward(self, x):
        x = F.one_hot(x)  # batch_size, rows, cols, onehot
        x = x.to(torch.float)
        x = x.permute([0, 3, 1, 2])  # batch_size, onehot, rows, cols
        return self.layers(x)
