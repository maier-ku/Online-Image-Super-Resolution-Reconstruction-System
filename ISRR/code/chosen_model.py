import math
from torch import nn
from models import ConvolutionalBlock, ResidualBlock, SubPixelConvolutionalBlock

# This file is for model implementation of different SRGAN models

class Generator(nn.Module):

    def __init__(self, large_kernel_size=9, small_kernel_size=3, n_channels=64, n_blocks=16, scaling_factor=4, model_selection=0):
        super(Generator, self).__init__()
        self.net = SRResNet(large_kernel_size=large_kernel_size, small_kernel_size=small_kernel_size,
                            n_channels=n_channels, n_blocks=n_blocks, scaling_factor=scaling_factor, model_selection=model_selection)

    def forward(self, lr_imgs):
        sr_imgs = self.net(lr_imgs)  # (N, n_channels, w * scaling factor, h * scaling factor)

        return sr_imgs


class SRResNet(nn.Module):
    """
    SRResNet
    """

    def __init__(self, large_kernel_size=9, small_kernel_size=3, n_channels=64, n_blocks=16, scaling_factor=4,
                 model_selection=0):
        super(SRResNet, self).__init__()

        scaling_factor = int(scaling_factor)
        assert scaling_factor in {2, 4, 8}, "scaling factor must be 2, 4 or 8!"
        self.model_selection = int(model_selection)
        assert model_selection in {0, 1, 2, 3, 4}, "model selection should be 0, 1, 2, 3, 4！"
        if self.model_selection == 3:
            self.dropout = nn.Dropout(0.3)
        else:
            self.dropout = nn.Dropout(0.2)

        self.conv_block1 = ConvolutionalBlock(in_channels=3, out_channels=n_channels, kernel_size=large_kernel_size,
                                              batch_norm=False, activation='PReLu')

        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(kernel_size=small_kernel_size, n_channels=n_channels) for i in range(n_blocks)])

        self.conv_block2 = ConvolutionalBlock(in_channels=n_channels, out_channels=n_channels,
                                              kernel_size=small_kernel_size,
                                              batch_norm=True, activation=None)

        n_subpixel_convolution_blocks = int(math.log2(scaling_factor))
        self.subpixel_convolutional_blocks = nn.Sequential(
            *[SubPixelConvolutionalBlock(kernel_size=small_kernel_size, n_channels=n_channels, scaling_factor=2) for i
              in range(n_subpixel_convolution_blocks)])

        self.conv_block3 = ConvolutionalBlock(in_channels=n_channels, out_channels=3, kernel_size=large_kernel_size,
                                              batch_norm=False, activation='Tanh')

    def forward(self, lr_imgs):
        output = self.conv_block1(lr_imgs)  # (16, 3, 24, 24)
        if self.model_selection == 2:
            output = self.dropout(output)
        residual = output  # (16, 64, 24, 24)
        output = self.residual_blocks(output)  # (16, 64, 24, 24)
        if self.model_selection == 1 or self.model_selection == 2:
            output = self.dropout(output)
        output = self.conv_block2(output)  # (16, 64, 24, 24)
        output = output + residual  # (16, 64, 24, 24)
        if self.model_selection == 0 or self.model_selection == 1 or self.model_selection == 2 or self.model_selection == 3:
            output = self.dropout(output)
        output = self.subpixel_convolutional_blocks(output)  # (16, 64, 24 * 4, 24 * 4)
        sr_imgs = self.conv_block3(output)  # (16, 3, 24 * 4, 24 * 4)

        return sr_imgs
