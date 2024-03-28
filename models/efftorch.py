import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet

class Classifier(nn.Module):
    def __init__(self, num_classes=22, efficientnet_version='b0'):
        super(Classifier, self).__init__()
        self.num_classes = num_classes
        self.model = EfficientNet.from_pretrained(f'efficientnet-{efficientnet_version}', num_classes=num_classes)

    def forward(self, x):
        x = self.model(x)
        return x