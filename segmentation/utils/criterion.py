import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossEntropyLoss2d(nn.Module):

    def __init__(self, weight=None):
        super().__init__()

        self.loss = nn.NLLLoss2d(weight)

    def forward(self, outputs, targets):
        return self.loss(F.log_softmax(outputs), targets)


class BinaryCrossEntropyLoss2d(nn.Module):

    def __init__(self, weight=None):
        super().__init__()
        
        #self.loss = nn.BCELoss(weight)
        self.loss = nn.BCEWithLogitsLoss(weight)

    def forward(self, outputs, targets):
        #return self.loss(nn.Sigmoid(outputs), targets)
        return self.loss(outputs, targets)