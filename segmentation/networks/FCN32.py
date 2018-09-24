import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from torch.utils import model_zoo
from torchvision import models

class FCN32(nn.Module):

    def __init__(self, num_classes):
        super().__init__()

        self.feats = models.vgg16(pretrained=True).features
        self.fconn = nn.Sequential(
            nn.Conv2d(512, 4096, 7),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Conv2d(4096, 4096, 1),
            nn.ReLU(inplace=True),
            nn.Dropout(),
        )
        self.score = nn.Conv2d(4096, num_classes, 1)

    def forward(self, x):
        feats = self.feats(x)
        fconn = self.fconn(feats)
        score = self.score(fconn)

        return F.upsample_bilinear(score, x.size()[2:])
