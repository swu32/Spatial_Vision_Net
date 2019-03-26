#!/usr/bin/env python3

import sys
import numpy as np
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo


class DivisiveNormalization(nn.Module):
    """Normalize input by dividing with normalization pool."""
    def __init__(self, requires_grad=False,
                 num_spatial_frequencies=8, num_orientations=8):
        super(DivisiveNormalization, self).__init__()

        self.num_features = num_spatial_frequencies * num_orientations
        self.num_spatial_frequencies = num_spatial_frequencies
        self.num_orientations = num_orientations

        # hard-coded parameters as in Schuett & Wichmann (2017)
        self.p = 2.0253
        self.q = 0.3676
        c = 0.0046
        self.c = torch.nn.Parameter(torch.Tensor([1.0]), requires_grad=False)
        #self.p = torch.nn.Parameter(torch.Tensor([self.p]), requires_grad=requires_grad) # not required
        #self.q = torch.nn.Parameter(torch.Tensor([self.q]), requires_grad=requires_grad) # not required
        #self.c = torch.nn.Parameter(torch.Tensor([self.c]), requires_grad=requires_grad) # not required
        # disentangling p and q in numerator and denominator of equation for flexibility
        self.p_plus_q = torch.nn.Parameter(torch.Tensor([self.p+self.q]), requires_grad=False)
        #self.c_pow_p  = torch.nn.Parameter(torch.Tensor([c**self.p]), requires_grad=True)
 
        # 1x1 convolution
        self.conv1x1 = nn.Conv2d(in_channels=self.num_features,
                                 out_channels=self.num_features,
                                 kernel_size=1, stride=1, padding=0,
                                 bias=False)
        # use only positive weights
        positive_weights = torch.abs(self.conv1x1.weight.data)
        self.conv1x1.weight = torch.nn.Parameter(positive_weights)


        # normalization weights: initialize as unitary matrix
        #self.weights = torch.rand(self.num_features, self.num_features)
        #for i in range(self.num_features):
        #    self.weights[i,i] = 1.0
        #self.b = torch.nn.Parameter(self.weights, requires_grad=True)
        #print(self.b.data.cpu().numpy())


    def forward(self, x):
        # input size: [batch_size, num_features, feature_size, feature_size]
        #        e.g. [128, 64, 112, 112]

        x_pow_p = torch.pow(x, self.p)
        if verbose:
            print("x_pow_p:")
            self.print_tensor(x_pow_p)

        # use absolute value of weights
        positive_weights = torch.abs(self.conv1x1.weight.data)
        self.conv1x1.weight = torch.nn.Parameter(positive_weights)

        conv1x1_output = self.conv1x1(x_pow_p) 
        if verbose:
            print("conv1x1_output:")
            self.print_tensor(conv1x1_output)

        numerator = torch.pow(x, self.p_plus_q)
        if verbose:
            print("numerator:")
            self.print_tensor(numerator)

        denominator = self.c + conv1x1_output
        if verbose:
            print("denominator:")
            self.print_tensor(denominator)
            print(np.abs(denominator.data.cpu().numpy()).min())

        result = torch.div(numerator, denominator)
        if verbose:
            print("result:")
            self.print_tensor(result)

        return result

    def print_tensor(self, x):
        print(self._p(x).min())
        print(self._p(x).max())
        print()

    def _p(self, x):
        return x.data.cpu().numpy()

