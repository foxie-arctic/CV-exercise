import torch
import torch.nn as nn


class TimeEmbeddingLayer(nn.Module):
    def __init__(self, model_dim, device):
        super(TimeEmbeddingLayer, self).__init__()
        self.model_dim = model_dim
        self.device = device

    def forward(self, timesteps):
        time_step_matrix = torch.reshape(timesteps, (-1, 1)).to(self.device)
        i_matrix = torch.pow(10000, torch.reshape(torch.arange(0, 2, self.model_dim), (1, -1)) / self.model_dim).to(
            self.device)
        time_embedding_table = torch.zeros(time_step_matrix.shape[0], self.model_dim).to(self.device)
        time_embedding_table[:, 0::2] = torch.sin(time_step_matrix / i_matrix)
        time_embedding_table[:, 1::2] = torch.cos(time_step_matrix / i_matrix)
        return time_embedding_table


class ConvTimeEmbedding(nn.Module):
    def __init__(self, ch_in, ch_time_emb, ch_out, kernel_size, stride, padding, bias=True, res_enable=False):
        super(ConvTimeEmbedding, self).__init__()
        self.res_enable = res_enable
        if self.res_enable is True:
            if ch_in == ch_out:
                self.ResLayer = nn.Identity()
            else:
                self.ResLayer = nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1, padding=0)
        self.ConvLayer = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            nn.BatchNorm2d(ch_out),
            nn.LeakyReLU())
        self.TimeEmbeddingLayer = nn.Sequential(nn.Linear(ch_time_emb, ch_out),
                                                nn.ReLU())

    def forward(self, X, time_emb):
        ConvLineX = self.ConvLayer(X)
        TimeEmb = self.TimeEmbeddingLayer(time_emb)[:, :, None, None]
        res = ConvLineX + TimeEmb
        if self.res_enable is True:
            assert res.shape[2] == X.shape[2], 'When applying res-connection, h_in should be equal to h_out'
            assert res.shape[3] == X.shape[3], 'When applying res-connection, w_in should be equal to w_out'
            res = res + self.ResLayer(X)
        else:
            res = res
        return res


class VanillaUNet_Encoder(nn.Module):
    def __init__(self, ch_in, ch_time_emb, mode='conv', res_enable=True):
        super(VanillaUNet_Encoder, self).__init__()
        self.mode = mode
        if mode == 'pool':
            self.down_pool_0 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)  # 64
            self.down_pool_1 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)  # 32
            self.down_pool_2 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)  # 16
            self.down_pool_3 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)  # 8
        elif mode == 'conv':
            self.down_pool_0 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
            self.down_pool_1 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1)
            self.down_pool_2 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)
            self.down_pool_3 = nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1)

        self.conv_0_0 = ConvTimeEmbedding(ch_in, ch_time_emb, 64, kernel_size=3, stride=1, padding=1,
                                          res_enable=res_enable)
        self.conv_0_1 = ConvTimeEmbedding(64, ch_time_emb, 64, kernel_size=3, stride=1, padding=1,
                                          res_enable=res_enable)

        self.conv_1_0 = ConvTimeEmbedding(64, ch_time_emb, 128, kernel_size=3, stride=1, padding=1,
                                          res_enable=res_enable)
        self.conv_1_1 = ConvTimeEmbedding(128, ch_time_emb, 128, kernel_size=3, stride=1, padding=1,
                                          res_enable=res_enable)

        self.conv_2_0 = ConvTimeEmbedding(128, ch_time_emb, 256, kernel_size=1, stride=1, padding=0)
        self.conv_2_1 = ConvTimeEmbedding(256, ch_time_emb, 256, kernel_size=3, stride=1, padding=1,
                                          res_enable=res_enable)

        self.conv_3_0 = ConvTimeEmbedding(256, ch_time_emb, 512, kernel_size=1, stride=1, padding=0)
        self.conv_3_1 = ConvTimeEmbedding(512, ch_time_emb, 512, kernel_size=3, stride=1, padding=1,
                                          res_enable=res_enable)

        self.conv_4_0 = ConvTimeEmbedding(512, ch_time_emb, 1024, kernel_size=1, stride=1, padding=0)
        self.conv_4_1 = ConvTimeEmbedding(1024, ch_time_emb, 1024, kernel_size=3, stride=1, padding=1,
                                          res_enable=res_enable)

    def forward(self, X, time_emb):
        his = []
        indices = []
        res = self.conv_0_0(X, time_emb)
        res = self.conv_0_1(res, time_emb)
        his.append(res)
        if self.mode == 'pool':
            res, idx0 = self.down_pool_0(res)
            indices.append(idx0)
        elif self.mode == 'conv':
            res = self.down_pool_0(res)
        res = self.conv_1_0(res, time_emb)
        res = self.conv_1_1(res, time_emb)
        his.append(res)
        if self.mode == 'pool':
            res, idx1 = self.down_pool_1(res)
            indices.append(idx1)
        elif self.mode == 'conv':
            res = self.down_pool_1(res)
        res = self.conv_2_0(res, time_emb)
        res = self.conv_2_1(res, time_emb)
        his.append(res)
        if self.mode == 'pool':
            res, idx2 = self.down_pool_0(res)
            indices.append(idx2)
        elif self.mode == 'conv':
            res = self.down_pool_2(res)
        res = self.conv_3_0(res, time_emb)
        res = self.conv_3_1(res, time_emb)
        his.append(res)
        if self.mode == 'pool':
            res, idx3 = self.down_pool_0(res)
            indices.append(idx3)
        elif self.mode == 'conv':
            res = self.down_pool_3(res)
        res = self.conv_4_0(res, time_emb)
        res = self.conv_4_1(res, time_emb)
        return res, his, indices


class VanillaUNet_Decoder(nn.Module):
    def __init__(self, ch_out, ch_time_emb, mode='conv', res_enable=True):
        super(VanillaUNet_Decoder, self).__init__()
        if mode == 'pool':
            self.up_pool_0 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=2, stride=2)
            self.up_pool_1 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2)
            self.up_pool_2 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2)
            self.up_pool_3 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2)
        elif mode == 'conv':
            self.up_pool_0 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=3, stride=2, padding=1,
                                                output_padding=1)
            self.up_pool_1 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=3, stride=2, padding=1,
                                                output_padding=1)
            self.up_pool_2 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1,
                                                output_padding=1)
            self.up_pool_3 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1,
                                                output_padding=1)

        self.conv_0_0 = ConvTimeEmbedding(1024, ch_time_emb, 512, kernel_size=1, stride=1, padding=0)
        self.conv_0_1 = ConvTimeEmbedding(512, ch_time_emb, 512, kernel_size=3, stride=1, padding=1,
                                          res_enable=res_enable)

        self.conv_1_0 = ConvTimeEmbedding(512, ch_time_emb, 256, kernel_size=1, stride=1, padding=0)
        self.conv_1_1 = ConvTimeEmbedding(256, ch_time_emb, 256, kernel_size=3, stride=1, padding=1,
                                          res_enable=res_enable)

        self.conv_2_0 = ConvTimeEmbedding(256, ch_time_emb, 128, kernel_size=3, stride=1, padding=1,
                                          res_enable=res_enable)
        self.conv_2_1 = ConvTimeEmbedding(128, ch_time_emb, 128, kernel_size=3, stride=1, padding=1,
                                          res_enable=res_enable)

        self.conv_3_0 = ConvTimeEmbedding(128, ch_time_emb, 64, kernel_size=3, stride=1, padding=1,
                                          res_enable=res_enable)
        self.conv_3_1 = ConvTimeEmbedding(64, ch_time_emb, 64, kernel_size=3, stride=1, padding=1,
                                          res_enable=res_enable)
        self.conv_3_2 = ConvTimeEmbedding(64, ch_time_emb, ch_out, kernel_size=3, stride=1, padding=1,
                                          res_enable=res_enable)

    def forward(self, X, time_emb, his):
        res = self.up_pool_0(X)
        res = torch.cat((his.pop(), res), dim=1)
        res = self.conv_0_0(res, time_emb)
        res = self.conv_0_1(res, time_emb)
        res = self.up_pool_1(res)
        res = torch.cat((his.pop(), res), dim=1)
        res = self.conv_1_0(res, time_emb)
        res = self.conv_1_1(res, time_emb)
        res = self.up_pool_2(res)
        res = torch.cat((his.pop(), res), dim=1)
        res = self.conv_2_0(res, time_emb)
        res = self.conv_2_1(res, time_emb)
        res = self.up_pool_3(res)
        res = torch.cat((his.pop(), res), dim=1)
        res = self.conv_3_0(res, time_emb)
        res = self.conv_3_1(res, time_emb)
        res = self.conv_3_2(res, time_emb)
        return res


class VanillaUNet(nn.Module):
    def __init__(self, ch_in=3, ch_time_emb=1024, ch_out=3, mode='conv', time_emb_device='cuda', res_enable=True):
        super(VanillaUNet, self).__init__()
        self.time_embedding_layer = nn.Sequential(TimeEmbeddingLayer(ch_time_emb, time_emb_device),
                                                  nn.Linear(ch_time_emb, ch_time_emb),
                                                  nn.SiLU(),
                                                  nn.Linear(ch_time_emb, ch_time_emb))
        self.encoder = VanillaUNet_Encoder(ch_in, ch_time_emb, mode=mode, res_enable=res_enable)
        self.decoder = VanillaUNet_Decoder(ch_out, ch_time_emb, mode=mode, res_enable=res_enable)

    def forward(self, X, timesteps):
        time_embedding_table = self.time_embedding_layer(timesteps)
        res, his, indices = self.encoder(X, time_embedding_table)
        res = self.decoder(res, time_embedding_table, his)
        return res
