import torch.nn as nn
import torch as th
from torch.nn import functional as F

# Return the flattened array
class View(nn.Module):
    def __init__(self,o):
        super(View, self).__init__()
        self.o = o
    def forward(self,x):
        return x.view(-1, self.o)

class Avg2d(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,x):
        sh = x.shape
        x = x.contiguous().view(sh[0], sh[1], -1)
        return x.mean(-1)

def num_parameters(model):
    return sum([w.numel() for w in model.parameters()])


class Id(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,x):
        return x

class Activation(nn.Module):
    def __init__(self):
        super().__init__()
        self.nonlinear = True

    def forward(self, x):
        y = F.relu(x) if self.nonlinear else x
        return y
