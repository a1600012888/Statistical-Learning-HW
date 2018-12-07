import torch
import torch.nn as nn
from typing import Tuple, List, Dict

class Affine(nn.Module):

    def __init__(self, inp_dim = 5000, out_dim = 1):
        super(Affine, self).__init__()

        self.fc = nn.Linear(inp_dim, out_dim, bias = False)

    def forward(self, inp):
        out = self.fc(inp)
        return out


def L1Penlty(net):

    loss = torch.norm(net.fc.weight, p = 1)
    return loss


class MultiStageLearningRatePolicy(object):
    '''
    '''

    _stages = None
    def __init__(self, stages:List[Tuple[int, float]]):

        assert(len(stages) >= 1)
        self._stages = stages


    def __call__(self, cur_ep:int) -> float:
        e = 0
        for pair in self._stages:
            e += pair[0]
            if cur_ep < e:
                return pair[1]
      #  return pair[-1][1]
        return pair[-1]

class LinearLearningRatePolicy(object):

    def __init__(self, max_iter, init_lr):
        self.max_iter = max_iter
        self.lr = init_lr

    def __call__(self, cur_iter:int) -> float:
        if cur_iter > self.max_iter:
            return 0
        a = cur_iter * 1.0 / self.max_iter
        lr = self.lr *(1 - a)

        return lr