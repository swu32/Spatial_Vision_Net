#!/usr/bin/env python3

import sys
import numpy as np
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo


#####################################################################
#   GABOR FILTERS
#####################################################################

def genGabor(size, frequency, orientation, func=np.cos, K=np.pi):
    radius = (int(size[0]/2.0), int(size[1]/2.0))
    [x, y] = np.meshgrid(range(-radius[0], radius[0]+1), range(-radius[1], radius[1]+1))

    x1 = x * np.cos(orientation) + y * np.sin(orientation)
    y1 = -x * np.sin(orientation) + y * np.cos(orientation)
    
    gauss = frequency**2 / (4*np.pi * K**2) * np.exp(- frequency**2 / (8*K**2) * ( 4 * x1**2 + y1**2))
    sinusoid = func(frequency * x1) * np.exp(K**2 / 2)
    gabor = gauss * sinusoid
    return gabor


def get_filter_banks(num_orientations=8,
                     num_spatial_frequencies=8,
                     size=64):

    orientations = np.arange(0, np.pi, np.pi/num_orientations)
    assert len(orientations) == num_orientations

    min_omega = 0.2
    max_omega = 1.0
    frequencies = np.arange(min_omega, max_omega, (max_omega-min_omega)/num_spatial_frequencies)
    assert len(frequencies) == num_spatial_frequencies

    sinFilterBank = []
    cosFilterBank = []
    gaborParams = []
    for f in frequencies:
        for o in orientations:
            gaborParam = {'frequency':f, 'orientation':o, 'size':(size, size)}
            sinGabor = genGabor(func=np.sin, **gaborParam)
            cosGabor = genGabor(func=np.cos, **gaborParam)
            sinFilterBank.append(sinGabor)
            cosFilterBank.append(cosGabor)
            gaborParams.append(gaborParam)

    return (sinFilterBank, cosFilterBank, gaborParams)


class GaborConvolution(nn.Module):
    """Convolves the input with Gabor filters."""
    def __init__(self, num_colour_channels, num_kernels=64, kernel_size=64, stride=2,
                 reflection_padding=0, convolution_padding=0,
                 bias=False, requires_grad=False):
        super(GaborConvolution, self).__init__()

        self.num_colour_channels = num_colour_channels
        self.num_kernels = num_kernels
        self.kernel_size = kernel_size
        self.reflection_padding=reflection_padding
        self.gabor_conv = nn.Conv2d(in_channels=self.num_colour_channels, out_channels=num_kernels,
                               kernel_size=kernel_size, stride=stride, padding=convolution_padding,
                               bias=bias)

        self._set_Gabor_parameters(requires_grad)
 
    def forward(self, x):
        reflection_pad = nn.ReflectionPad2d(self.reflection_padding)
        x = reflection_pad(x)
        x = self.gabor_conv(x)

        return x


    def _set_Gabor_parameters(self, requires_grad):

        sin_filter_bank, cos_filter_bank, _ = get_filter_banks()
        assert self.gabor_conv.weight.size()[0] == len(cos_filter_bank)
        assert len(sin_filter_bank) == len(cos_filter_bank)

        new_filter_weights = torch.ones(self.num_kernels, self.num_colour_channels,
                                        self.kernel_size, self.kernel_size)

        for i in range(len(cos_filter_bank)):
            for z in range(self.num_colour_channels):
                cos_filter_value_vector = cos_filter_bank[i]
                new_filter_weights[i,z,:,:] = torch.Tensor(
                    np.square(cos_filter_value_vector)[:self.kernel_size, :self.kernel_size])
        
        self.gabor_conv.weight = torch.nn.Parameter(new_filter_weights)
        self.gabor_conv.weight.requires_grad = requires_grad


