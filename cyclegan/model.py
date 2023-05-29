#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim import lr_scheduler
import numpy as np


# In[ ]:


def define_discriminator(image_shape):
    # Weight initialization
    def weights_init_normal(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                init.constant_(m.bias.data, 0.0)


    # Define the discriminator network
    model = nn.Sequential(
        # C64: 4x4 kernel Stride 2x2
        nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
        nn.LeakyReLU(0.2, inplace=True),
        # C128: 4x4 kernel Stride 2x2
        nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
        nn.InstanceNorm2d(128),
        nn.LeakyReLU(0.2, inplace=True),
        # C256: 4x4 kernel Stride 2x2
        nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
        nn.InstanceNorm2d(256),
        nn.LeakyReLU(0.2, inplace=True),
        # C512: 4x4 kernel Stride 2x2
        nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
        nn.InstanceNorm2d(512),
        nn.LeakyReLU(0.2, inplace=True),
        # Second last output layer: 4x4 kernel Stride 1x1
        nn.Conv2d(512, 512, kernel_size=4, stride=1, padding=1),
        nn.InstanceNorm2d(512),
        nn.LeakyReLU(0.2, inplace=True),
        # Patch output
        nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)
    )

    # Initialize the weights
    model.apply(weights_init_normal)

    return model


# In[ ]:


def resnet_block(n_filters, input_layer):
    # Weight initialization
    def weights_init_normal(m):
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find("InstanceNorm2d") != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0.0)

    # Define the residual block
    model = nn.Sequential(
        # First convolutional layer
        nn.Conv2d(n_filters, n_filters, kernel_size=3, stride=1, padding=1),
        nn.InstanceNorm2d(n_filters),
        nn.ReLU(inplace=True),
        # Second convolutional layer
        nn.Conv2d(n_filters, n_filters, kernel_size=3, stride=1, padding=1),
        nn.InstanceNorm2d(n_filters)
    )

    # Initialize the weights
    model.apply(weights_init_normal)

    # Concatenate merge channel-wise with input layer
    return nn.ReLU(inplace=True)(torch.cat([model(input_layer), input_layer], dim=1))


# In[ ]:


def define_generator(image_shape, n_resnet=1):
    # Weight initialization
    def weights_init_normal(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                init.constant_(m.bias.data, 0.0)


    # Define the residual block
    class ResidualBlock(nn.Module):
        def __init__(self, n_filters):
            super(ResidualBlock, self).__init__()

            self.conv1 = nn.Conv2d(n_filters, n_filters, kernel_size=3, stride=1, padding=1)
            self.norm1 = nn.InstanceNorm2d(n_filters)
            self.conv2 = nn.Conv2d(n_filters, n_filters, kernel_size=3, stride=1, padding=1)
            self.norm2 = nn.InstanceNorm2d(n_filters)

        def forward(self, x):
            residual = x

            out = F.relu(self.norm1(self.conv1(x)))
            out = self.norm2(self.conv2(out))

            out = out + residual

            return out

    # Define the generator network
    class Generator(nn.Module):
        def __init__(self, image_shape, n_resnet):
            super(Generator, self).__init__()

            self.c7s1_64 = nn.Sequential(
                nn.ReflectionPad2d(3),
                nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=0),
                nn.InstanceNorm2d(64),
                nn.ReLU(inplace=True)
            )

            self.d128 = nn.Sequential(
                nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                nn.InstanceNorm2d(128),
                nn.ReLU(inplace=True)
            )

            self.d256 = nn.Sequential(
                nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
                nn.InstanceNorm2d(256),
                nn.ReLU(inplace=True)
            )

            self.resnet_blocks = nn.ModuleList([ResidualBlock(256) for _ in range(n_resnet)])

            self.u128 = nn.Sequential(
                nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(128),
                nn.ReLU(inplace=True)
            )

            self.u64 = nn.Sequential(
                nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(64),
                nn.ReLU(inplace=True)
            )

            self.c7s1_3 = nn.Sequential(
                nn.ReflectionPad2d(3),
                nn.Conv2d(64, 3, kernel_size=7, stride=1, padding=0),
                nn.InstanceNorm2d(3),
                nn.Tanh()
            )

        def forward(self, x):
            out = self.c7s1_64(x)
            out = self.d128(out)
            out = self.d256(out)

            for resnet_block in self.resnet_blocks:
                out = resnet_block(out)

            out = self.u128(out)
            out = self.u64(out)
            out = self.c7s1_3(out)

            return out

    # Create an instance of the generator
    generator = Generator(image_shape, n_resnet)

    # Initialize the weights
    generator.apply(weights_init_normal)

    return generator


# In[ ]:


def define_composite_model(g_model_1, d_model, g_model_2, image_shape):

   

    g_model_1.trainable = True
    # mark discriminator and second generator as non-trainable
    d_model.trainable = False
    g_model_2.trainable = False

    # Define the composite model
    class CompositeModel(nn.Module):
        def __init__(self, g_model_1, d_model, g_model_2):
            super(CompositeModel, self).__init__()

            self.g_model_1 = g_model_1
            self.d_model = d_model
            self.g_model_2 = g_model_2

        def forward(self, input_gen, input_id):
            gen1_out = self.g_model_1(input_gen)
            output_d = self.d_model(gen1_out)

            output_id = self.g_model_1(input_id)

            output_f = self.g_model_2(gen1_out)

            gen2_out = self.g_model_2(input_id)
            output_b = self.g_model_1(gen2_out)

            return output_d, output_id, output_f, output_b

    # Create an instance of the composite model
    composite_model = CompositeModel(g_model_1, d_model, g_model_2)

    # Define the optimizer
    opt = optim.Adam(composite_model.parameters(), lr=0.001, betas=(0.5, 0.999))

    return composite_model, opt


# In[ ]:




