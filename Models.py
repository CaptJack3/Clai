''' Here I will Place the Neural Network Environment'''
import math
import numpy as np
import enum
import torch
import torch.nn as nn
import torch.nn.functional as F

class Actions(enum.Enum):
    Skip = 0
    Buy = 1
    Close = 2

DEFAULT_BARS_COUNT = 10
DEFAULT_COMMISSION_PERC = 0.1

class DQNConv1D(nn.Module):
    def __init__(self, shape, actions_n):
        super(DQNConv1D, self).__init__()

        ''' Define adjustable parameters'''
        N1 = 8  # Was 128
        N2 = 15  # Was 512 In the assignment
        ker_size = 1  # Was 5

        self.conv = nn.Sequential(
            nn.Conv1d(shape[0], N1, ker_size),
            nn.ReLU(),
            nn.Conv1d(N1, N1, ker_size),
            nn.ReLU(),
        )

        out_size = self._get_conv_out(shape)

        self.fc_val = nn.Sequential(
            nn.Linear(out_size, N2),
            nn.ReLU(),
            nn.Linear(N2, 1)
        )

        self.fc_adv = nn.Sequential(
            nn.Linear(out_size, N2),
            nn.ReLU(),
            nn.Linear(N2, actions_n)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        conv_out = self.conv(x).view(x.size()[0], -1)
        val = self.fc_val(conv_out)
        adv = self.fc_adv(conv_out)
        return val + adv - adv.mean(dim=1, keepdim=True)