import torch
import torch.nn as nn


class SmoothL1Loss(nn.Module):

    def __init__(self):
        super(SmoothL1Loss, self).__init__()

    def forward(self, preds, targets):
        """
        计算边界框回归损失。N表示RoI个数
        :param preds: 大小为[N, 4] x,y,w,h
        :param targets: 大小为[N, 4] x,y,w,h
        :return:
        """
        res = self.smooth_l1(preds - targets)
        return torch.sum(res)

    def smooth_l1(self, x):
        if torch.abs(x) < 1:
            return 0.5 * torch.pow(x, 2)
        else:
            return torch.abs(x) - 0.5