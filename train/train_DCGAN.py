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
from models import DCGAN
from tqdm import *
import time

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

first_time = False
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


generator = DCGAN.DCGAN_Generator(latent_dim=128).to(device)
discriminator = DCGAN.DCGAN_Discriminator(latent_dim=128).to(device)

if not first_time:
    generator.load_state_dict(torch.load('../saved_model/DCGAN/GNet.pth'))
    discriminator.load_state_dict(torch.load('../saved_model/DCGAN/DNet.pth'))
    generator.eval()
    discriminator.eval()

criterion_GAN = torch.nn.BCELoss()

optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))


def train_loop(dataloader, latent_dim, train_device):
    generator.train()
    discriminator.train()
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        X_dev = X.to(train_device)
        batch_size = X_dev.shape[0]

        valid_label = torch.ones((batch_size, 1), requires_grad=False).to(train_device)
        fake_label = torch.zeros((batch_size, 1), requires_grad=False).to(train_device)

        optimizer_G.zero_grad()
        z = torch.randn((batch_size, latent_dim), device=train_device)
        G_res = generator(z)
        G_loss = criterion_GAN(discriminator(G_res), valid_label)
        G_loss.backward()
        optimizer_G.step()

        optimizer_D.zero_grad()
        D_valid_loss = criterion_GAN(discriminator(X_dev), valid_label)
        D_fake_loss = criterion_GAN(discriminator(G_res.detach()), fake_label)
        D_loss = D_valid_loss + D_fake_loss
        D_loss.backward()
        optimizer_D.step()

        if batch % 100 == 0:
            g_loss, d_loss, current = G_loss.item(), D_loss.item(), batch * len(X)
            print(f"g_loss: {g_loss:>7f} d_loss: {d_loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(generate_number, latent_dim, test_device):
    generator.eval()
    discriminator.eval()
    z = torch.randn((generate_number, latent_dim), device=test_device)
    res = generator(z)

    for j in range(generate_number):
        recon = res.permute(0, 2, 3, 1).detach().cpu().numpy()[j]
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
    train_loop(train_data, 128, device)
    torch.save(generator.state_dict(), '../saved_model/DCGAN/GNet.pth')
    torch.save(discriminator.state_dict(), '../saved_model/DCGAN/DNet.pth')
    test_loop(3, 128, device)
print("Done!")
test_loop(5, 128, device)
