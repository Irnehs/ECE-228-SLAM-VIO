import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F

# This will be the overarching class that connects all 4 of our NNs
class SLAMPathError(nn.Modulee):
    def __init__(self):
        super(SLAMPathError, self).__init__()

    def forward(self, x):
        pass