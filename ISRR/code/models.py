import torch
from torch import nn
import torchvision
import math


class ConvolutionalBlock(nn.Module):
    """
    Convolutional module, consisting of a convolutional layer, a BN normalization layer, and an activation layer.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, batch_norm=False, activation=None):
        """
        :in_channels: Number of input channels
        :out_channels: Number of output channels
        :kernel_size
        :stride
        :batch_norm: Whether BN layer is included
        :activation: Activation layer type; None if not
        """
        super(ConvolutionalBlock, self).__init__()

        if activation is not None:
            activation = activation.lower()
            assert activation in {'prelu', 'leakyrelu', 'tanh'}

        # List of layers
        layers = list()

        # 1 convolutional layer
        layers.append(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                      padding=kernel_size // 2))

        # 1 BN normalised layer
        if batch_norm is True:
            layers.append(nn.BatchNorm2d(num_features=out_channels))

        # 1 activation level
        if activation == 'prelu':
            layers.append(nn.PReLU())
        elif activation == 'leakyrelu':
            layers.append(nn.LeakyReLU(0.2))
        elif activation == 'tanh':
            layers.append(nn.Tanh())

        # Merge layer
        self.conv_block = nn.Sequential(*layers)

    def forward(self, input):
        """
        Forward propagation

        :input: set of input images, tensor representation, size (N, in_channels, w, h)
        :return: output image set, tensor representation, size (N, out_channels, w, h)
        """
        output = self.conv_block(input)

        return output


class SubPixelConvolutionalBlock(nn.Module):
    """
    Sub-pixel convolution module, containing convolution, pixel cleaning and activation layers.
    """

    def __init__(self, kernel_size=3, n_channels=64, scaling_factor=2):
        super(SubPixelConvolutionalBlock, self).__init__()

        # The number of channels is first expanded by convolution to a scaling factor^2 times
        self.conv = nn.Conv2d(in_channels=n_channels, out_channels=n_channels * (scaling_factor ** 2),
                              kernel_size=kernel_size, padding=kernel_size // 2)
        # Perform pixel cleaning and merge relevant channel data
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor=scaling_factor)
        # Finally add the activation layer
        self.prelu = nn.PReLU()

    def forward(self, input):
        """
        Forward propagation.

        :input: size (N, n_channels, w, h)
        :output: size (N, n_channels, w * scaling factor, h * scaling factor)
        """
        output = self.conv(input)  # (N, n_channels * scaling factor^2, w, h)
        output = self.pixel_shuffle(output)  # (N, n_channels, w * scaling factor, h * scaling factor)
        output = self.prelu(output)  # (N, n_channels, w * scaling factor, h * scaling factor)

        return output


class ResidualBlock(nn.Module):
    """
    Residual module, consisting of two convolution modules and a skip connection .
    """

    def __init__(self, kernel_size=3, n_channels=64):
        """
        :kernel_size
        :n_channels: parameter n_channels: number of input and output channels (the number of input and output channels is the same since it is a ResNet network and needs to be skipped)
        """
        super(ResidualBlock, self).__init__()

        # First convolutional block
        self.conv_block1 = ConvolutionalBlock(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size,
                                              batch_norm=True, activation='PReLu')

        # Second convolutional block
        self.conv_block2 = ConvolutionalBlock(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size,
                                              batch_norm=True, activation=None)

    def forward(self, input):
        residual = input  # (N, n_channels, w, h)
        output = self.conv_block1(input)  # (N, n_channels, w, h)
        output = self.conv_block2(output)  # (N, n_channels, w, h)
        output = output + residual  # (N, n_channels, w, h)

        return output


class SRResNet(nn.Module):
    """
    SRResNet模型
    """
    def __init__(self, large_kernel_size=9, small_kernel_size=3, n_channels=64, n_blocks=16, scaling_factor=4):
        """
        :参数 large_kernel_size: First layer of convolution and last layer of convolution kernel size
        :参数 small_kernel_size: Intermediate layer convolutional kernel size
        :参数 n_channels: Number of intermediate level channels
        :参数 n_blocks: Number of residual modules
        :参数 scaling_factor: scaling_factor
        """
        super(SRResNet, self).__init__()

        # factor scaling must be 2, 4 or 8
        scaling_factor = int(scaling_factor)
        assert scaling_factor in {2, 4, 8}, "放大比例必须为 2、 4 或 8!"

        # First convolutional block
        self.conv_block1 = ConvolutionalBlock(in_channels=3, out_channels=n_channels, kernel_size=large_kernel_size,
                                              batch_norm=False, activation='PReLu')

        # A series of residual modules, each containing a skip connection
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(kernel_size=small_kernel_size, n_channels=n_channels) for i in range(n_blocks)])

        # Second convolutional block
        self.conv_block2 = ConvolutionalBlock(in_channels=n_channels, out_channels=n_channels,
                                              kernel_size=small_kernel_size,
                                              batch_norm=True, activation=None)

        # Enlargement is achieved by sub-pixel convolution modules, each of which is enlarged by a factor of two
        n_subpixel_convolution_blocks = int(math.log2(scaling_factor))
        self.subpixel_convolutional_blocks = nn.Sequential(
            *[SubPixelConvolutionalBlock(kernel_size=small_kernel_size, n_channels=n_channels, scaling_factor=2) for i
              in range(n_subpixel_convolution_blocks)])

        self.dropout = nn.Dropout(0.2)
        # Last convolutional block
        self.conv_block3 = ConvolutionalBlock(in_channels=n_channels, out_channels=3, kernel_size=large_kernel_size,
                                              batch_norm=False, activation='Tanh')

    def forward(self, lr_imgs):
        """
        Forward propagation.

        :lr_imgs: Number lr_imgs: set of low-resolution input images, tensor representation, size (N, 3, w, h)
        :返回: 高分辨率输出图像集, 张量表示， 大小为 (N, 3, w * scaling factor, h * scaling factor)
        """
        output = self.conv_block1(lr_imgs)  # (16, 3, 24, 24)
        output = self.residual_blocks(output)  # (16, 64, 24, 24)
        output = self.conv_block2(output)  # (16, 64, 24, 24)
        output = self.subpixel_convolutional_blocks(output)  # (16, 64, 24 * 4, 24 * 4)
        sr_imgs = self.conv_block3(output)  # (16, 3, 24 * 4, 24 * 4)

        return sr_imgs


class Generator(nn.Module):
    """
    generator model, whose structure is identical to that of SRResNet.
    """

    def __init__(self, large_kernel_size=9, small_kernel_size=3, n_channels=64, n_blocks=16, scaling_factor=4):
        super(Generator, self).__init__()
        self.net = SRResNet(large_kernel_size=large_kernel_size, small_kernel_size=small_kernel_size,
                            n_channels=n_channels, n_blocks=n_blocks, scaling_factor=scaling_factor)

    def forward(self, lr_imgs):
        """
        : Forward propagation.

        参数 lr_imgs: Low precision image (N, 3, w, h)
        返回: Super-resolution Image (N, 3, w * scaling factor, h * scaling factor)
        """
        sr_imgs = self.net(lr_imgs)  # (N, n_channels, w * scaling factor, h * scaling factor)

        return sr_imgs


class Discriminator(nn.Module):

    def __init__(self, kernel_size=3, n_channels=64, n_blocks=8, fc_size=1024):
        super(Discriminator, self).__init__()
        in_channels = 3
        # Convolution series, designed with reference to the paper SRGAN
        conv_blocks = list()
        for i in range(n_blocks):
            out_channels = (n_channels if i is 0 else in_channels * 2) if i % 2 is 0 else in_channels
            conv_blocks.append(
                ConvolutionalBlock(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                   stride=1 if i % 2 is 0 else 2, batch_norm=i is not 0, activation='LeakyReLu'))
            in_channels = out_channels
        self.conv_blocks = nn.Sequential(*conv_blocks)

        # Fixed output size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((6, 6))

        self.fc1 = nn.Linear(out_channels * 6 * 6, fc_size)

        self.leaky_relu = nn.LeakyReLU(0.2)

        self.fc2 = nn.Linear(1024, 1)

        # Finally there is no need to add a sigmoid layer, as PyTorch's nn.BCEWithLogitsLoss() already includes this step

    def forward(self, imgs):
        """
        Forward propagation.

        参数 imgs: The original HD image or the super-resolution reconstructed image to be used as a discriminator, expressed as a tensor of size (N, 3, w * scaling factor, h * scaling factor)
        返回: a score to determine whether an image is an HD image, tensor, size (N)
        """
        batch_size = imgs.size(0)
        output = self.conv_blocks(imgs)
        output = self.adaptive_pool(output)
        output = self.fc1(output.view(batch_size, -1))
        output = self.leaky_relu(output)
        logit = self.fc2(output)

        return logit


class TruncatedVGG19(nn.Module):
    """
    truncated VGG19 network for computing MSE loss in VGG feature space
    """

    def __init__(self, i, j):
        """
        :i: The i-th pooling layer
        :j: The j-th convolutional layer
        """
        super(TruncatedVGG19, self).__init__()

        # Load the pre-trained VGG model
        vgg19 = torchvision.models.vgg19(pretrained=True)  # C:\Users\Administrator/.cache\torch\checkpoints\vgg19-dcbb9e9d.pth

        maxpool_counter = 0
        conv_counter = 0
        truncate_at = 0
        # Iterative search
        for layer in vgg19.features.children():
            truncate_at += 1

            # Statistics
            if isinstance(layer, nn.Conv2d):
                conv_counter += 1
            if isinstance(layer, nn.MaxPool2d):
                maxpool_counter += 1
                conv_counter = 0

            # Truncate the jth convolutional layer after the (i-1)th pooling layer (before the i-th pooling layer)
            if maxpool_counter == i - 1 and conv_counter == j:
                break

        # Check if conditions are met
        assert maxpool_counter == i - 1 and conv_counter == j, "当前 i=%d 、 j=%d 不满足 VGG19 模型结构" % (
            i, j)

        # Truncated network structure
        self.truncated_vgg19 = nn.Sequential(*list(vgg19.features.children())[:truncate_at + 1])

    def forward(self, input):
        """
        Forward propagation
        input: HD original or super-resolution reconstructed map, tensor representation, size (N, 3, w * scaling factor, h * scaling factor)
        VGG19 feature map, tensor representation, size (N, feature_map_channels, feature_map_w, feature_map_h)
        """
        output = self.truncated_vgg19(input)  # (N, feature_map_channels, feature_map_w, feature_map_h)

        return output
