import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from torch import einsum


class SVM(nn.Module):
    def __init__(self, bins, num_classes):
        super().__init__()
        self.svm = nn.Sequential(
                       nn.Linear(bins, num_classes),
                  )


    def criterion(self, pred, label):
        return torch.sum(torch.clamp(1 - pred.t() * label, min=0))


    def forward(self, x, label):
        pred = self.svm(x)
        loss = self.criterion(pred, label)
        return pred.view(-1), loss
