import torch
import torch.nn as nn


class Sinh(nn.Module):
    def forward(self, x):
        return torch.sinh(x).detach()


class Cosh(nn.Module):
    def forward(self, x):
        return torch.cosh(x).detach()
