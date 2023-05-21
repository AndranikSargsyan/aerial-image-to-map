# original: https://github.com/NVIDIA/pix2pixHD/blob/master/models/networks.py
from collections import defaultdict

import numpy as np
import torch.nn as nn

from qartezator.models.modules.base import BaseDiscriminator


class DotDict(defaultdict):
    # https://stackoverflow.com/questions/2352181/how-to-use-a-dot-to-access-members-of-dictionary
    """dot.notation access to dictionary attributes"""
    __getattr__ = defaultdict.get
    __setattr__ = defaultdict.__setitem__
    __delattr__ = defaultdict.__delitem__

class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class MultidilatedResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, conv_layer, norm_layer, activation=nn.ReLU(True), use_dropout=False):
        super().__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, conv_layer, norm_layer, activation, use_dropout)

    def build_conv_block(self, dim, padding_type, conv_layer, norm_layer, activation, use_dropout, dilation=1):
        conv_block = []
        conv_block += [conv_layer(dim, dim, kernel_size=3, padding_mode=padding_type),
                       norm_layer(dim),
                       activation]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        conv_block += [conv_layer(dim, dim, kernel_size=3, padding_mode=padding_type),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(BaseDiscriminator):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d,):
        super().__init__()
        self.n_layers = n_layers

        kw = 4
        padw = int(np.ceil((kw-1.0)/2))
        sequence = [[nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
                     nn.LeakyReLU(0.2, True)]]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)

            cur_model = []
            cur_model += [
                nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
                norm_layer(nf),
                nn.LeakyReLU(0.2, True)
            ]
            sequence.append(cur_model)

        nf_prev = nf
        nf = min(nf * 2, 512)

        cur_model = []
        cur_model += [
            nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
            norm_layer(nf),
            nn.LeakyReLU(0.2, True)
        ]
        sequence.append(cur_model)

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        for n in range(len(sequence)):
            setattr(self, 'model'+str(n), nn.Sequential(*sequence[n]))

    def get_all_activations(self, x):
        res = [x]
        for n in range(self.n_layers + 2):
            model = getattr(self, 'model' + str(n))
            res.append(model(res[-1]))
        return res[1:]

    def forward(self, x):
        act = self.get_all_activations(x)
        return act[-1], act[:-1]
