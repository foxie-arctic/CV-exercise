import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torchvision import transforms
from torch.utils.data import DataLoader
from torch import optim

import matplotlib.pyplot as plt
import random

from datasets import CelebA
from models import UNet_DDPM
from models import VanillaUNet
from tqdm import *
import time

first_time = True
model_type = 'Vanilla'


def diffusion(x_0, beta_start, beta_end, timesteps, t, diffusion_device):
    x_0 = torch.tensor(x_0).to(diffusion_device)
    betas = torch.linspace(beta_start, beta_end, timesteps)
    '''
    start start+(end-start)/(steps-1) ... end
    len == steps
    '''
    alphas = 1 - betas
    alphas_bars = torch.cumprod(alphas, dim=0).to(diffusion_device)
    alpha_bar = alphas_bars[t]
    eps = torch.randn_like(x_0)
    x_t = torch.sqrt(alpha_bar) * x_0 + torch.sqrt(1 - alpha_bar) * eps
    return x_t


def reverse_diffusion(model, x_t, beta_start, beta_end, timesteps, reverse_device):
    betas = torch.linspace(beta_start, beta_end, timesteps).to(reverse_device)
    alphas = 1 - betas
    alphas_bars = torch.cumprod(alphas, dim=0)
    alphas_cumprod_prev = F.pad(alphas_bars[:-1], pad=(1, 0), value=1.)
    # :-1是取所有除去最后一个元素 ::-1是取倒序
    res = []
    for t in range(timesteps - 1, -1, -1):
        alphas_bar_t_1 = alphas_cumprod_prev[t]
        alphas_bar_t_1 = alphas_bar_t_1.reshape(-1, 1, 1, 1)
        alphas_bar_t = alphas_bars[t]
        alphas_bar_t = alphas_bar_t.reshape(-1, 1, 1, 1)
        beta_t = betas[t]
        beta_t = beta_t.reshape(-1, 1, 1, 1)
        alpha_t = alphas[t]
        alpha_t = alpha_t.reshape(-1, 1, 1, 1)
        sigma = torch.sqrt(((1 - alphas_bar_t_1) * beta_t) / (1 - alphas_bar_t))
        z = torch.randn_like(x_t).to(reverse_device)
        t_tensor = torch.tensor([t]).to(reverse_device)
        with torch.no_grad():
            scaled_noise_pred_mu = beta_t / torch.sqrt(1 - alphas_bar_t) * model(x_t, t_tensor)
        if t > 0:
            x_t_1 = 1 / torch.sqrt(alpha_t) * (x_t - scaled_noise_pred_mu) + sigma * z
        else:
            x_t_1 = 1 / torch.sqrt(alpha_t) * (x_t - scaled_noise_pred_mu)
        x_t = x_t_1
        '''
        if t % 100 == 0:
            res.append((x_t.permute(0, 2, 3, 1).cpu().numpy()[0]))
        '''
        if t == 0:
            res = x_t.permute(0, 2, 3, 1).cpu().numpy()[0]
    return res


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

image_transform = transforms.Compose([transforms.ToPILImage(),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.CenterCrop(148),
                                      transforms.Resize([128, 128]),
                                      transforms.ToTensor()])
train_set = CelebA.CelebADataset('../data/img_align_celeba/img_align_celeba/', is_train=True, transform=image_transform,
                                 target_transform=None)
test_set = CelebA.CelebADataset('../data/img_align_celeba/img_align_celeba/', is_train=False, transform=image_transform,
                                target_transform=None)
train_data = DataLoader(train_set, batch_size=32, shuffle=True)
test_data = DataLoader(test_set, batch_size=32, shuffle=False)

for i in range(10):  # [0,3]
    ax = plt.subplot(2, 5, i + 1)  # subplot: 可划分的画布
    idx = random.randint(0, len(train_set)-1)
    digit_0 = train_set[idx][0].clone()  # 复制测试集tensor数据到digit_0
    digit_0 = digit_0.permute(1, 2, 0)  # 给matlib显示需要H W C
    digit_0_image = digit_0.reshape(128, 128, 3)  # 将打平的数据转换为image形式 128*128*3
    ax.imshow(digit_0_image, interpolation='nearest')  # 将内容加入画布
    ax.set_title('label: {}'.format(train_set[idx][1]), fontsize=10, color='black')  # 标题为：label: label string/num
plt.show()  # 显示画布

if model_type == 'DDPM':
    noise_prediction_model = UNet_DDPM.UNet_DDPM(ch_in=3, ch_mid=128, ch_time_emb=512, ch_out=3, group=16,
                                                 drop_rate=0.5,
                                                 device=device).to(device)
elif model_type == 'Vanilla':
    noise_prediction_model = VanillaUNet.VanillaUNet(ch_in=3, ch_time_emb=1024, ch_out=3, time_emb_device=device,
                                                     mode='conv', res_enable=True).to(device)
else:
    noise_prediction_model = None

if not first_time:
    noise_prediction_model.load_state_dict(torch.load('../saved_model/DDPM/DDPM.pth'))
    noise_prediction_model.eval()
print(noise_prediction_model)

# criterion_DDPM = nn.SmoothL1Loss()
criterion_DDPM = nn.MSELoss()
optimizer_DDPM = optim.Adam(noise_prediction_model.parameters(), lr=5e-4)


def train_loop(dataloader, model, loss_fn, optimizer, beta_start, beta_end, timesteps, train_device):
    model.train()
    size = len(dataloader.dataset)
    betas = torch.linspace(beta_start, beta_end, timesteps).to(train_device)
    alphas = 1 - betas
    alphas_bars = torch.cumprod(alphas, dim=0)
    for batch, (X, y) in enumerate(dataloader):
        X_dev = X.to(train_device)
        X_dev = 2 * X_dev - 1
        batch_size = X_dev.shape[0]
        t = torch.randint(0, timesteps, [batch_size]).to(train_device)
        alphas_bar = alphas_bars.gather(-1, t).reshape(batch_size, 1, 1, 1)
        eps = torch.randn_like(X_dev)
        X_t = torch.sqrt(alphas_bar) * X_dev + torch.sqrt(1 - alphas_bar) * eps
        noise_pred = model(X_t, t)
        loss = loss_fn(noise_pred, eps)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(model, beta_start, beta_end, timesteps, test_device):
    model.eval()
    print('---------------')
    print('reverse process')
    print('---------------')
    random_sample = torch.randn(1, 3, 128, 128).to(test_device)
    recon = reverse_diffusion(model, random_sample, beta_start, beta_end, timesteps, test_device)
    recon_min = np.min(recon, axis=(0, 1), keepdims=True)
    recon_max = np.max(recon, axis=(0, 1), keepdims=True)

    recon = (recon - recon_min) / (recon_max - recon_min)
    recon = recon.clip(0, 1)
    plot_ax = plt.subplot(1, 1, 1)
    plot_ax.imshow(recon, interpolation='nearest')
    plt.show()


epochs = 100
for train_num in range(epochs):
    print(f"Epoch {train_num + 1}\n-------------------------------")
    train_loop(train_data, noise_prediction_model, criterion_DDPM, optimizer_DDPM, 0.0001, 0.02, 1000, device)
    torch.save(noise_prediction_model.state_dict(), '../saved_model/DDPM/DDPM.pth')
    test_loop(noise_prediction_model, 0.0001, 0.02, 1000, device)
print("Done!")
test_loop(noise_prediction_model, 0.0001, 0.02, 1000, device)
