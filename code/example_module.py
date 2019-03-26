#!/usr/bin/env python3

import sys
import numpy as np
import torch
import torch.nn as nn

# example code for a new module
class NewModule(nn.Module):
    def __init__(self):
        super(NewModule, self).__init__()

    def forward(self, x):
        pass


