from __future__ import division
import ipdb
import inspect
import os
import time
import math
import glob
import numpy as np
from six.moves import xrange
import pickle
import sys
import config as cfg
import torch.nn as nn
from torch.autograd import Variable
import math
import torch
# torch.backends.cudnn.enabled=False

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import torchvision.models as models

# Load pre-trained VGG-16 model
model_vgg16 = models.vgg16(pretrained=True)

class Classifier(nn.Module):

    def __init__(self, num_classes=22):
        super(Classifier, self).__init__()
        self.num_classes = num_classes
        self._initialize_weights()

    def _initialize_weights(self):
        self.model = models.vgg16(num_classes=self.num_classes)
        print("[OK] Weights initialized randomly.")

    def forward(self, x):
        x = self.model(x)
        return x