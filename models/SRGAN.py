import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, ch_in):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(nn.Conv2d(ch_in, ch_in, kernel_size=3, stride=1, padding=1),
                                   nn.BatchNorm2d(ch_in),
                                   nn.PReLU(),
                                   nn.Conv2d(ch_in, ch_in, kernel_size=3, stride=1, padding=1),
                                   nn.BatchNorm2d(ch_in))

    def forward(self, X):
        res = self.block(X)
        return res + X


class SubPixelConvBlock(nn.Module):
    def __init__(self, ch_in, r):
        super(SubPixelConvBlock, self).__init__()
        self.block = nn.Sequential(nn.Conv2d(ch_in, ch_in * (r ** 2), kernel_size=3, stride=1, padding=1),
                                   nn.PixelShuffle(r),
                                   nn.PReLU())

    def forward(self, X):
        res = self.block(X)
        return res


class BatchNormalizationConvBlock(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, stride, padding):
        super(BatchNormalizationConvBlock, self).__init__()
        self.block = nn.Sequential(nn.Conv2d(ch_in, ch_out, kernel_size=kernel_size, stride=stride, padding=padding),
                                   nn.BatchNorm2d(ch_out),
                                   nn.LeakyReLU())

    def forward(self, X):
        res = self.block(X)
        return res


class SRGAN_Generator(nn.Module):
    def __init__(self, ch_in):
        super(SRGAN_Generator, self).__init__()
        self.feature_extraction = nn.Sequential(nn.Conv2d(ch_in, 64, kernel_size=9, stride=1, padding=4),
                                                nn.PReLU())
        self.residual_blocks = nn.Sequential(ResidualBlock(64),
                                             ResidualBlock(64),
                                             ResidualBlock(64),
                                             ResidualBlock(64),
                                             ResidualBlock(64))
        self.conv_before_skip = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                                              nn.BatchNorm2d(64))
        self.up_sample = nn.Sequential(SubPixelConvBlock(64, 2),
                                       SubPixelConvBlock(64, 2))
        self.final_construction = nn.Conv2d(64, 3, kernel_size=9, stride=1, padding=4)

    def forward(self, LR):
        res = self.feature_extraction(LR)
        skip_connection = res
        res = self.residual_blocks(res)
        res = self.conv_before_skip(res)
        res = res + skip_connection
        res = self.up_sample(res)
        SR = self.final_construction(res)
        return SR


class SRGAN_Discriminator(nn.Module):
    def __init__(self, ch_in):
        super(SRGAN_Discriminator, self).__init__()
        self.feature_extraction = nn.Sequential(nn.Conv2d(ch_in, 64, kernel_size=3, stride=1, padding=1),
                                                nn.PReLU())
        self.BN_conv_blocks = nn.Sequential(BatchNormalizationConvBlock(64, 64, kernel_size=3, stride=2, padding=1),
                                            BatchNormalizationConvBlock(64, 128, kernel_size=3, stride=1, padding=1),
                                            BatchNormalizationConvBlock(128, 128, kernel_size=3, stride=2, padding=1),
                                            BatchNormalizationConvBlock(128, 256, kernel_size=3, stride=1, padding=1),
                                            BatchNormalizationConvBlock(256, 256, kernel_size=3, stride=2, padding=1),
                                            BatchNormalizationConvBlock(256, 512, kernel_size=3, stride=1, padding=1),
                                            BatchNormalizationConvBlock(512, 512, kernel_size=3, stride=2, padding=1))
        self.fc_classification = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                               nn.Flatten(),
                                               nn.Linear(512, 1024),
                                               nn.LeakyReLU(),
                                               nn.Linear(1024, 1),
                                               nn.Sigmoid())

    def forward(self, SHR):
        res = self.feature_extraction(SHR)
        res = self.BN_conv_blocks(res)
        res = self.fc_classification(res)
        return res

    
