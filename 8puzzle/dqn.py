import numpy as np
from torch import nn
from torch.nn import functional as F

from config import *

class DQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.Linear((GRID_SIZE ** 2) ** 2, 256),
            nn.Linear(256, 256),
            nn.Linear(256, 256),
            nn.Linear(256, 4),
        )
    def forward(self, x):
        x = F.one_hot(x)
        x = x.view(x.size(0), -1)
        return self.main(x.float())