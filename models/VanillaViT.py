import torch
import torch.nn as nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import torch.nn.functional as F
import math


class VanillaViT(nn.Module):
    def __init__(self, ch_in=3, image_size=32, patch_size=4, model_dim=512, dim_ff=512, nhead=8,
                 num_layers=6, num_classes=10, dropout_rate=0.1, trans_dropout=0.1):
        super(VanillaViT, self).__init__()
        self.patch_embedding_layer = nn.Conv2d(in_channels=ch_in, out_channels=model_dim, kernel_size=patch_size,
                                               stride=patch_size)
        self.class_token_embedding = nn.Parameter(torch.randn(1, 1, model_dim), requires_grad=True)
        self.patch_num = (image_size // patch_size) ** 2
        self.position_embedding = nn.Parameter(torch.randn(1, self.patch_num + 1, model_dim), requires_grad=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.encoder_layer = nn.TransformerEncoderLayer(model_dim, nhead=nhead, dim_feedforward=dim_ff,
                                                        dropout=trans_dropout)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.MLP = nn.Sequential(nn.LayerNorm(model_dim),
                                 nn.Linear(model_dim, num_classes))

    def forward(self, X):
        patch_embedding = self.patch_embedding_layer(X)
        batch, oc, oh, ow = patch_embedding.shape
        patch_embedding = torch.reshape(patch_embedding, (batch, oc, -1))
        patch_embedding = torch.permute(patch_embedding, (0, 2, 1))

        class_token_embedding = repeat(self.class_token_embedding, '() n d -> b n d', b=batch)
        token_embedding = torch.cat((class_token_embedding, patch_embedding), dim=1)

        token_embedding = token_embedding + self.position_embedding
        token_embedding = self.dropout(token_embedding)
        encoder_output = self.encoder(token_embedding)

        class_token_output = encoder_output[:, 0, :]
        res = self.MLP(class_token_output)

        return res
