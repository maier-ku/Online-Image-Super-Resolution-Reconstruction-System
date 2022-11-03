import math

from torch import nn

from models.models import ConvolutionalBlock, ResidualBlock, SubPixelConvolutionalBlock


class Generator(nn.Module):
    """
    生成器模型，其结构与SRResNet完全一致.
    """

    def __init__(self, large_kernel_size=9, small_kernel_size=3, n_channels=64, n_blocks=16, scaling_factor=4, model_selection=4):
        """
        参数 large_kernel_size：第一层和最后一层卷积核大小
        参数 small_kernel_size：中间层卷积核大小
        参数 n_channels：中间层卷积通道数
        参数 n_blocks: 残差模块数量
        参数 scaling_factor: 放大比例
        """
        super(Generator, self).__init__()
        self.net = SRResNet(large_kernel_size=large_kernel_size, small_kernel_size=small_kernel_size,
                            n_channels=n_channels, n_blocks=n_blocks, scaling_factor=scaling_factor, model_selection=model_selection)

    def forward(self, lr_imgs):
        """
        前向传播.

        参数 lr_imgs: 低精度图像 (N, 3, w, h)
        返回: 超分重建图像 (N, 3, w * scaling factor, h * scaling factor)
        """
        sr_imgs = self.net(lr_imgs)  # (N, n_channels, w * scaling factor, h * scaling factor)

        return sr_imgs


class SRResNet(nn.Module):
    """
    SRResNet模型
    """

    def __init__(self, large_kernel_size=9, small_kernel_size=3, n_channels=64, n_blocks=16, scaling_factor=4,
                 model_selection=4):
        """
        :参数 large_kernel_size: 第一层卷积和最后一层卷积核大小
        :参数 small_kernel_size: 中间层卷积核大小
        :参数 n_channels: 中间层通道数
        :参数 n_blocks: 残差模块数
        :参数 scaling_factor: 放大比例
        """
        super(SRResNet, self).__init__()

        # 放大比例必须为 2、 4 或 8
        scaling_factor = int(scaling_factor)
        assert scaling_factor in {2, 4, 8}, "放大比例必须为 2、 4 或 8!"
        self.model_selection = int(model_selection)
        assert model_selection in {0, 1, 2, 3, 4}, "模型必须为0、1、2、3、4！"
        if self.model_selection == 3:
            self.dropout = nn.Dropout(0.3)
        else:
            self.dropout = nn.Dropout(0.2)

        # 第一个卷积块
        self.conv_block1 = ConvolutionalBlock(in_channels=3, out_channels=n_channels, kernel_size=large_kernel_size,
                                              batch_norm=False, activation='PReLu')

        # 一系列残差模块, 每个残差模块包含一个跳连接
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(kernel_size=small_kernel_size, n_channels=n_channels) for i in range(n_blocks)])

        # 第二个卷积块
        self.conv_block2 = ConvolutionalBlock(in_channels=n_channels, out_channels=n_channels,
                                              kernel_size=small_kernel_size,
                                              batch_norm=True, activation=None)

        # 放大通过子像素卷积模块实现, 每个模块放大两倍
        n_subpixel_convolution_blocks = int(math.log2(scaling_factor))
        self.subpixel_convolutional_blocks = nn.Sequential(
            *[SubPixelConvolutionalBlock(kernel_size=small_kernel_size, n_channels=n_channels, scaling_factor=2) for i
              in range(n_subpixel_convolution_blocks)])

        # 最后一个卷积模块
        self.conv_block3 = ConvolutionalBlock(in_channels=n_channels, out_channels=3, kernel_size=large_kernel_size,
                                              batch_norm=False, activation='Tanh')

    def forward(self, lr_imgs):
        """
        前向传播.

        :参数 lr_imgs: 低分辨率输入图像集, 张量表示，大小为 (N, 3, w, h)
        :返回: 高分辨率输出图像集, 张量表示， 大小为 (N, 3, w * scaling factor, h * scaling factor)
        # 加了3个dropout层
        """
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
