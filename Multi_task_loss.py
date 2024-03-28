import torch
import torch.nn as nn

from smooth_l1_loss import SmoothL1Loss


class MultiTaskLoss(nn.Module):

    def __init__(self, lam=1):
        super(MultiTaskLoss, self).__init__()
        self.lam = lam
        self.cls = nn.CrossEntropyLoss()
        self.loc = SmoothL1Loss()

    def forward(self, scores, preds, targets):
        """
        :param scores: [N, C]
        :param preds: [N, 4]x,y,w,h
        :param targets: [N]
        :return:
        """
        N = targets.shape[0]
        return self.cls(scores, targets) + self.loc(scores[range(N), self.indicator(targets)],
                                                    preds[range(N), self.indicator(preds)])

    def indicator(self, cate):
        return cate != 0