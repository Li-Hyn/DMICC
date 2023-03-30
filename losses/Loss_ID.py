import torch.nn as nn
import torch.nn.functional as F

class Loss_ID(nn.Module):
    def __init__(self, tau2):
        super().__init__()
        self.tau2 = tau2

    def forward(self, x, y):

        L_id = F.cross_entropy(x, y)


        return L_id