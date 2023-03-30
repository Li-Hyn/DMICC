import torch
import torch.nn as nn
import sys

class Loss_FMI(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, ff):

        norm_fx = ff / (ff ** 2).sum(0, keepdim=True).sqrt()
        coef_mat = torch.mm(norm_fx.t(), norm_fx)
        k = coef_mat.size(0)
        lamb = 10
        EPS = sys.float_info.epsilon
        p_i = coef_mat.sum(dim=1).view(k, 1).expand(k, k)
        p_j = coef_mat.sum(dim=0).view(1, k).expand(k, k)
        p_i_j = torch.where(coef_mat < EPS, torch.tensor([EPS], device=coef_mat.device), coef_mat)
        p_j = torch.where(p_j < EPS, torch.tensor([EPS], device=p_j.device), p_j)
        p_i = torch.where(p_i < EPS, torch.tensor([EPS], device=p_i.device), p_i)

        loss_fmi = (p_i_j * (torch.log(p_i_j) \
                          - (lamb + 1) * torch.log(p_j) \
                          - (lamb + 1) * torch.log(p_i))) / (k**2)

        loss_fmi = loss_fmi.sum()

        return loss_fmi