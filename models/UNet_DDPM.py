import torch
import torch.nn as nn
import math


class TimeEmbeddingResBlock(nn.Module):
    def __init__(self, ch_in, ch_time_emb, ch_out, group=32, drop_rate=0.5):
        super(TimeEmbeddingResBlock, self).__init__()

        self.TimeLine = nn.Sequential(nn.SiLU(),
                                      nn.Linear(ch_time_emb, ch_out))

        if ch_in == ch_out:
            self.ResLine = nn.Identity()
        else:
            self.ResLine = nn.Conv2d(ch_in, ch_out, kernel_size=1, bias=True)

        self.RedimConvLine = nn.Sequential(nn.GroupNorm(group, ch_in),
                                           nn.SiLU(),
                                           nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True))

        self.MainConvLine = nn.Sequential(nn.GroupNorm(group, ch_out),
                                          nn.SiLU(),
                                          nn.Dropout(drop_rate),
                                          nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True))

    def forward(self, ipt, time_embedding):

        ResIpt = self.ResLine(ipt)
        RedimIpt = self.RedimConvLine(ipt)
        RedimTimeEmb = self.TimeLine(time_embedding)[:, :, None, None]
        #   HIGHLIGHT!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        MainConv = self.MainConvLine(RedimTimeEmb + RedimIpt)
        opt = MainConv + ResIpt
        return opt


class SelfAttentionResBlock(nn.Module):
    def __init__(self, ch_in, ch_out, group=32):
        super(SelfAttentionResBlock, self).__init__()
        self.MainGroupNorm = nn.GroupNorm(group, ch_in)
        self.QLine = nn.Conv2d(ch_in, ch_out, kernel_size=1, bias=True)
        self.KLine = nn.Conv2d(ch_in, ch_out, kernel_size=1, bias=True)
        self.VLine = nn.Conv2d(ch_in, ch_out, kernel_size=1, bias=True)
        self.AttentionScore2Prob = nn.Softmax(dim=-1)
        if ch_in == ch_out:
            self.ResLine = nn.Identity()
        else:
            self.ResLine = nn.Conv2d(ch_in, ch_out, kernel_size=1, bias=True)

    def forward(self, ipt):
        ResIpt = self.ResLine(ipt)
        GroupNormIpt = self.MainGroupNorm(ipt)
        RawQ = self.QLine(GroupNormIpt)
        B, C_out, H, W = RawQ.shape
        Q_Transpose = torch.reshape(RawQ, (B, C_out, -1))
        Q = torch.permute(Q_Transpose, (0, 2, 1))
        RawK = self.KLine(GroupNormIpt)
        K_Transpose = torch.reshape(RawK, (B, C_out, -1))
        AttentionScore = torch.bmm(Q, K_Transpose) / math.sqrt(C_out)
        # HIGHLIGHT!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        AttentionProb = self.AttentionScore2Prob(AttentionScore)
        RawV = self.VLine(GroupNormIpt)
        V_Transpose = torch.reshape(RawV, (B, C_out, -1))
        V = torch.permute(V_Transpose, (0, 2, 1))
        RawRelativity = torch.bmm(AttentionProb, V)
        Relativity = torch.reshape(torch.permute(RawRelativity, (0, 2, 1)), (B, C_out, H, W))
        opt = ResIpt + Relativity
        return opt


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


class UNet_DDPM_Encoder(nn.Module):
    def __init__(self, ch_in=3, ch_mid=128, ch_time_emb=512, group=32, drop_rate=0.5):
        super(UNet_DDPM_Encoder, self).__init__()
        self.RedimConv_ch_in_ch_mid = nn.Conv2d(ch_in, ch_mid, kernel_size=3, stride=1, padding=1, bias=True)
        self.ResBlock_ch_mid_0 = TimeEmbeddingResBlock(ch_mid, ch_time_emb, ch_mid, group, drop_rate)
        self.ResBlock_ch_mid_1 = TimeEmbeddingResBlock(ch_mid, ch_time_emb, ch_mid, group, drop_rate)
        self.ResizeConv_Down_0 = nn.Conv2d(ch_mid, ch_mid, kernel_size=3, stride=2, padding=1, bias=True)
        self.ResBlock_ch_mid_2 = TimeEmbeddingResBlock(ch_mid, ch_time_emb, ch_mid, group, drop_rate)
        self.ResBlock_ch_mid_3 = TimeEmbeddingResBlock(ch_mid, ch_time_emb, ch_mid, group, drop_rate)
        self.ResizeConv_Down_1 = nn.Conv2d(ch_mid, ch_mid, kernel_size=3, stride=2, padding=1, bias=True)
        self.ResBlock_ch_mid_2ch_mid = TimeEmbeddingResBlock(ch_mid, ch_time_emb, 2 * ch_mid, group, drop_rate)
        self.ResBlock_2ch_mid_0 = TimeEmbeddingResBlock(2 * ch_mid, ch_time_emb, 2 * ch_mid, group, drop_rate)
        self.ResizeConv_Down_2 = nn.Conv2d(2 * ch_mid, 2 * ch_mid, kernel_size=3, stride=2, padding=1, bias=True)
        self.ResBlock_2ch_mid_1 = TimeEmbeddingResBlock(2 * ch_mid, ch_time_emb, 2 * ch_mid, group, drop_rate)
        self.AttnBlock_2ch_mid_0 = SelfAttentionResBlock(2 * ch_mid, 2 * ch_mid, group)
        self.ResBlock_2ch_mid_2 = TimeEmbeddingResBlock(2 * ch_mid, ch_time_emb, 2 * ch_mid, group, drop_rate)
        self.AttnBlock_2ch_mid_1 = SelfAttentionResBlock(2 * ch_mid, 2 * ch_mid, group)
        self.ResizeConv_Down_3 = nn.Conv2d(2 * ch_mid, 2 * ch_mid, kernel_size=3, stride=2, padding=1, bias=True)
        self.ResBlock_2ch_mid_4ch_mid = TimeEmbeddingResBlock(2 * ch_mid, ch_time_emb, 4 * ch_mid, group, drop_rate)
        self.ResBlock_4ch_mid_0 = TimeEmbeddingResBlock(4 * ch_mid, ch_time_emb, 4 * ch_mid, group, drop_rate)
        self.ResizeConv_Down_4 = nn.Conv2d(4 * ch_mid, 4 * ch_mid, kernel_size=3, stride=2, padding=1, bias=True)
        self.ResBlock_4ch_mid_1 = TimeEmbeddingResBlock(4 * ch_mid, ch_time_emb, 4 * ch_mid, group, drop_rate)
        self.ResBlock_4ch_mid_2 = TimeEmbeddingResBlock(4 * ch_mid, ch_time_emb, 4 * ch_mid, group, drop_rate)

    def forward(self, ipt, time_emb):
        level_res = []
        res = self.RedimConv_ch_in_ch_mid(ipt)
        level_res.append(res)
        res = self.ResBlock_ch_mid_0(res, time_emb)
        level_res.append(res)
        res = self.ResBlock_ch_mid_1(res, time_emb)
        level_res.append(res)
        res = self.ResizeConv_Down_0(res)
        level_res.append(res)
        res = self.ResBlock_ch_mid_2(res, time_emb)
        level_res.append(res)
        res = self.ResBlock_ch_mid_3(res, time_emb)
        level_res.append(res)
        res = self.ResizeConv_Down_1(res)
        level_res.append(res)
        res = self.ResBlock_ch_mid_2ch_mid(res, time_emb)
        level_res.append(res)
        res = self.ResBlock_2ch_mid_0(res, time_emb)
        level_res.append(res)
        res = self.ResizeConv_Down_2(res)
        level_res.append(res)
        res = self.ResBlock_2ch_mid_1(res, time_emb)
        res = self.AttnBlock_2ch_mid_0(res)
        level_res.append(res)
        res = self.ResBlock_2ch_mid_2(res, time_emb)
        res = self.AttnBlock_2ch_mid_1(res)
        level_res.append(res)
        res = self.ResizeConv_Down_3(res)
        level_res.append(res)
        res = self.ResBlock_2ch_mid_4ch_mid(res, time_emb)
        level_res.append(res)
        res = self.ResBlock_4ch_mid_0(res, time_emb)
        level_res.append(res)
        res = self.ResizeConv_Down_4(res)
        level_res.append(res)
        res = self.ResBlock_4ch_mid_1(res, time_emb)
        level_res.append(res)
        res = self.ResBlock_4ch_mid_2(res, time_emb)
        level_res.append(res)
        return res, level_res


class UNet_DDPM_LatentCoder(nn.Module):
    def __init__(self, ch_mid=128, ch_time_emb=512, group=32, drop_rate=0.5):
        super(UNet_DDPM_LatentCoder, self).__init__()
        self.ResBlock_4ch_mid_0 = TimeEmbeddingResBlock(4 * ch_mid, ch_time_emb, 4 * ch_mid, group, drop_rate)
        self.AttnBlock_4ch_mid_0 = SelfAttentionResBlock(4 * ch_mid, 4 * ch_mid, group)
        self.ResBlock_4ch_mid_1 = TimeEmbeddingResBlock(4 * ch_mid, ch_time_emb, 4 * ch_mid, group, drop_rate)

    def forward(self, ipt, time_emb):
        res = self.ResBlock_4ch_mid_0(ipt, time_emb)
        res = self.AttnBlock_4ch_mid_0(res)
        res = self.ResBlock_4ch_mid_1(res, time_emb)
        return res


class UNet_DDPM_Decoder(nn.Module):
    def __init__(self, ch_mid=128, ch_out=3, ch_time_emb=512, group=32, drop_rate=0.5):
        super(UNet_DDPM_Decoder, self).__init__()
        self.ResBlock_8ch_mid_4ch_mid_0 = TimeEmbeddingResBlock(8 * ch_mid, ch_time_emb, 4 * ch_mid, group, drop_rate)
        self.ResBlock_8ch_mid_4ch_mid_1 = TimeEmbeddingResBlock(8 * ch_mid, ch_time_emb, 4 * ch_mid, group, drop_rate)
        self.ResBlock_8ch_mid_4ch_mid_2 = TimeEmbeddingResBlock(8 * ch_mid, ch_time_emb, 4 * ch_mid, group, drop_rate)
        self.ResizeConv_Up_0 = nn.ConvTranspose2d(4 * ch_mid, 4 * ch_mid, kernel_size=3, stride=2, padding=1,
                                                  output_padding=1, bias=True)
        self.ResBlock_8ch_mid_4ch_mid_3 = TimeEmbeddingResBlock(8 * ch_mid, ch_time_emb, 4 * ch_mid, group, drop_rate)
        self.ResBlock_8ch_mid_4ch_mid_4 = TimeEmbeddingResBlock(8 * ch_mid, ch_time_emb, 4 * ch_mid, group, drop_rate)
        self.ResBlock_6ch_mid_4ch_mid_0 = TimeEmbeddingResBlock(6 * ch_mid, ch_time_emb, 4 * ch_mid, group, drop_rate)
        self.ResizeConv_Up_1 = nn.ConvTranspose2d(4 * ch_mid, 4 * ch_mid, kernel_size=3, stride=2, padding=1,
                                                  output_padding=1, bias=True)
        self.ResBlock_6ch_mid_2ch_mid_0 = TimeEmbeddingResBlock(6 * ch_mid, ch_time_emb, 2 * ch_mid, group, drop_rate)
        self.AttnBlock_2ch_mid_0 = SelfAttentionResBlock(2 * ch_mid, 2 * ch_mid, group)
        self.ResBlock_4ch_mid_2ch_mid_0 = TimeEmbeddingResBlock(4 * ch_mid, ch_time_emb, 2 * ch_mid, group, drop_rate)
        self.AttnBlock_2ch_mid_1 = SelfAttentionResBlock(2 * ch_mid, 2 * ch_mid, group)
        self.ResBlock_4ch_mid_2ch_mid_1 = TimeEmbeddingResBlock(4 * ch_mid, ch_time_emb, 2 * ch_mid, group, drop_rate)
        self.AttnBlock_2ch_mid_2 = SelfAttentionResBlock(2 * ch_mid, 2 * ch_mid, group)
        self.ResizeConv_Up_2 = nn.ConvTranspose2d(2 * ch_mid, 2 * ch_mid, kernel_size=3, stride=2, padding=1,
                                                  output_padding=1, bias=True)
        self.ResBlock_4ch_mid_2ch_mid_2 = TimeEmbeddingResBlock(4 * ch_mid, ch_time_emb, 2 * ch_mid, group, drop_rate)
        self.ResBlock_4ch_mid_2ch_mid_3 = TimeEmbeddingResBlock(4 * ch_mid, ch_time_emb, 2 * ch_mid, group, drop_rate)
        self.ResBlock_3ch_mid_2ch_mid_0 = TimeEmbeddingResBlock(3 * ch_mid, ch_time_emb, 2 * ch_mid, group, drop_rate)
        self.ResizeConv_Up_3 = nn.ConvTranspose2d(2 * ch_mid, 2 * ch_mid, kernel_size=3, stride=2, padding=1,
                                                  output_padding=1, bias=True)
        self.ResBlock_3ch_mid_ch_mid_0 = TimeEmbeddingResBlock(3 * ch_mid, ch_time_emb, ch_mid, group, drop_rate)
        self.ResBlock_2ch_mid_ch_mid_0 = TimeEmbeddingResBlock(2 * ch_mid, ch_time_emb, ch_mid, group, drop_rate)
        self.ResBlock_2ch_mid_ch_mid_1 = TimeEmbeddingResBlock(2 * ch_mid, ch_time_emb, ch_mid, group, drop_rate)
        self.ResizeConv_Up_4 = nn.ConvTranspose2d(ch_mid, ch_mid, kernel_size=3, stride=2, padding=1,
                                                  output_padding=1, bias=True)
        self.ResBlock_2ch_mid_ch_mid_2 = TimeEmbeddingResBlock(2 * ch_mid, ch_time_emb, ch_mid, group, drop_rate)
        self.ResBlock_2ch_mid_ch_mid_3 = TimeEmbeddingResBlock(2 * ch_mid, ch_time_emb, ch_mid, group, drop_rate)
        self.ResBlock_2ch_mid_ch_mid_4 = TimeEmbeddingResBlock(2 * ch_mid, ch_time_emb, ch_mid, group, drop_rate)
        self.OutputLayer = nn.Sequential(nn.GroupNorm(group, ch_mid),
                                         nn.SiLU(),
                                         nn.Conv2d(ch_mid, ch_out, kernel_size=3, stride=1, padding=1, bias=True))

    def forward(self, ipt, time_emb, level_res):
        res = self.ResBlock_8ch_mid_4ch_mid_0(torch.cat([ipt, level_res.pop()], dim=1), time_emb)
        res = self.ResBlock_8ch_mid_4ch_mid_1(torch.cat([res, level_res.pop()], dim=1), time_emb)
        res = self.ResBlock_8ch_mid_4ch_mid_2(torch.cat([res, level_res.pop()], dim=1), time_emb)
        res = self.ResizeConv_Up_0(res)
        res = self.ResBlock_8ch_mid_4ch_mid_3(torch.cat([res, level_res.pop()], dim=1), time_emb)
        res = self.ResBlock_8ch_mid_4ch_mid_4(torch.cat([res, level_res.pop()], dim=1), time_emb)
        res = self.ResBlock_6ch_mid_4ch_mid_0(torch.cat([res, level_res.pop()], dim=1), time_emb)
        res = self.ResizeConv_Up_1(res)
        res = self.ResBlock_6ch_mid_2ch_mid_0(torch.cat([res, level_res.pop()], dim=1), time_emb)
        res = self.AttnBlock_2ch_mid_0(res)
        res = self.ResBlock_4ch_mid_2ch_mid_0(torch.cat([res, level_res.pop()], dim=1), time_emb)
        res = self.AttnBlock_2ch_mid_1(res)
        res = self.ResBlock_4ch_mid_2ch_mid_1(torch.cat([res, level_res.pop()], dim=1), time_emb)
        res = self.AttnBlock_2ch_mid_2(res)
        res = self.ResizeConv_Up_2(res)
        res = self.ResBlock_4ch_mid_2ch_mid_2(torch.cat([res, level_res.pop()], dim=1), time_emb)
        res = self.ResBlock_4ch_mid_2ch_mid_3(torch.cat([res, level_res.pop()], dim=1), time_emb)
        res = self.ResBlock_3ch_mid_2ch_mid_0(torch.cat([res, level_res.pop()], dim=1), time_emb)
        res = self.ResizeConv_Up_3(res)
        res = self.ResBlock_3ch_mid_ch_mid_0(torch.cat([res, level_res.pop()], dim=1), time_emb)
        res = self.ResBlock_2ch_mid_ch_mid_0(torch.cat([res, level_res.pop()], dim=1), time_emb)
        res = self.ResBlock_2ch_mid_ch_mid_1(torch.cat([res, level_res.pop()], dim=1), time_emb)
        res = self.ResizeConv_Up_4(res)
        res = self.ResBlock_2ch_mid_ch_mid_2(torch.cat([res, level_res.pop()], dim=1), time_emb)
        res = self.ResBlock_2ch_mid_ch_mid_3(torch.cat([res, level_res.pop()], dim=1), time_emb)
        res = self.ResBlock_2ch_mid_ch_mid_4(torch.cat([res, level_res.pop()], dim=1), time_emb)
        res = self.OutputLayer(res)
        return res


class UNet_DDPM(nn.Module):
    def __init__(self, ch_in=3, ch_mid=128, ch_time_emb=512, ch_out=3, group=32, drop_rate=0.5, device='cuda'):
        super(UNet_DDPM, self).__init__()
        self.encoder = UNet_DDPM_Encoder(ch_in, ch_mid, ch_time_emb, group, drop_rate)
        self.latent_coder = UNet_DDPM_LatentCoder(ch_mid, ch_time_emb, group, drop_rate)
        self.decoder = UNet_DDPM_Decoder(ch_mid, ch_out, ch_time_emb, group, drop_rate)

        self.time_embedding_layer = nn.Sequential(TimeEmbeddingLayer(ch_mid, device),
                                                  nn.Linear(ch_mid, 4 * ch_mid),
                                                  nn.SiLU(),
                                                  nn.Linear(4 * ch_mid, 4 * ch_mid))

    def forward(self, ipt, time_table):
        time_embedding_table = self.time_embedding_layer(time_table)
        res, his_res = self.encoder(ipt, time_embedding_table)
        res = self.latent_coder(res, time_embedding_table)
        res = self.decoder(res, time_embedding_table, his_res)

        return res
