import torch
import torch.nn as nn


class ConvRes(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, stride, padding, bias=True, res_enable=False,
                 activation='leaky_relu'):
        super(ConvRes, self).__init__()
        self.res_enable = res_enable
        if self.res_enable is True:
            if ch_in == ch_out:
                self.ResLayer = nn.Identity()
            else:
                self.ResLayer = nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1, padding=0)
        if activation == 'leaky_relu':
            self.ConvLayer = nn.Sequential(
                nn.Conv2d(ch_in, ch_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
                nn.BatchNorm2d(ch_out),
                nn.LeakyReLU())
        elif activation == 'tanh':
            self.ConvLayer = nn.Sequential(
                nn.Conv2d(ch_in, ch_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
                nn.BatchNorm2d(ch_out),
                nn.Tanh())

    def forward(self, X):
        res = self.ConvLayer(X)
        if self.res_enable is True:
            assert res.shape[2] == X.shape[2], 'When applying res-connection, h_in should be equal to h_out'
            assert res.shape[3] == X.shape[3], 'When applying res-connection, w_in should be equal to w_out'
            res = res + self.ResLayer(X)
        else:
            res = res
        return res


class ConvTransposeBlock(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, stride, padding, output_padding, activation='leaky_relu'):
        super(ConvTransposeBlock, self).__init__()
        if activation == 'leaky_relu':
            self.ConvTransposeLayer = nn.Sequential(
                nn.ConvTranspose2d(ch_in, ch_out, kernel_size=kernel_size, stride=stride, padding=padding,
                                   output_padding=output_padding),
                nn.BatchNorm2d(ch_out),
                nn.LeakyReLU())
        elif activation == 'tanh':
            self.ConvTransposeLayer = nn.Sequential(
                nn.ConvTranspose2d(ch_in, ch_out, kernel_size=kernel_size, stride=stride, padding=padding,
                                   output_padding=output_padding),
                nn.BatchNorm2d(ch_out),
                nn.Tanh())

    def forward(self, X):
        res = self.ConvTransposeLayer(X)
        return res


class DCGAN_Generator(nn.Module):
    def __init__(self, latent_dim=128):
        super(DCGAN_Generator, self).__init__()
        self.latent2img = nn.Sequential(nn.Linear(latent_dim, 512 * 2 * 2),
                                        nn.LeakyReLU())
        self.img_generate_layer = nn.Sequential(ConvTransposeBlock(512, 256, 3, 2, 1, 1),
                                                ConvTransposeBlock(256, 128, 3, 2, 1, 1),
                                                ConvTransposeBlock(128, 64, 3, 2, 1, 1),
                                                ConvTransposeBlock(64, 32, 3, 2, 1, 1),
                                                ConvTransposeBlock(32, 16, 3, 2, 1, 1),
                                                ConvTransposeBlock(16, 3, 3, 2, 1, 1, 'tanh'))

    def forward(self, z):
        res = self.latent2img(z)
        res = torch.reshape(res, (-1, 512, 2, 2))
        res = self.img_generate_layer(res)
        return res


class DCGAN_Discriminator(nn.Module):
    def __init__(self, latent_dim=128):
        super(DCGAN_Discriminator, self).__init__()
        self.ConvLayer = nn.Sequential(
            ConvRes(3, 16, 3, 2, 1),
            ConvRes(16, 32, 3, 2, 1),
            ConvRes(32, 64, 3, 2, 1),
            ConvRes(64, 128, 3, 2, 1),
            ConvRes(128, 256, 3, 2, 1),
            ConvRes(256, 512, 3, 2, 1)
        )
        self.JudgeLayer = nn.Sequential(nn.Flatten(),
                                        nn.Linear(512 * 2 * 2, latent_dim),
                                        nn.LeakyReLU(),
                                        nn.Linear(latent_dim, 1),
                                        nn.Sigmoid())

    def forward(self, X):
        res = self.ConvLayer(X)
        res = self.JudgeLayer(res)
        return res
