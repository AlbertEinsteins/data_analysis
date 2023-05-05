# conv classifier

import torch
from torch import nn
from torchvision.models import resnet34


class NNClassifier(nn.Module):
    def __init__(self, n_classes=10):
        super().__init__()
        self.classifier = resnet34(pretrained=True, num_classes=n_classes)


    def forward(self, x):
        return self.classifier(x)

