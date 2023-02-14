import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torchvision import transforms
from torch.utils.data import DataLoader
from torch import optim

from PIL import Image
import matplotlib.pyplot as plt
import random

from datasets import RealSR
from models import SRGAN
from torchvision.models.vgg import vgg16
from tqdm import *
import time


class GeneratorLoss(nn.Module):
    def __init__(self, judge_device, pixel_weight=1, feature_weight=0.006, TV_weight=2e-8, adversarial_weight=0.001):
        super(GeneratorLoss, self).__init__()

        self.pixel_loss = nn.MSELoss()
        self.feature_loss = VGG16_pretrained_feature_extractor(judge_device)
        self.TV_loss = TV_loss()
        self.adversarial_loss = nn.BCELoss()

        self.pixel_weight = pixel_weight
        self.feature_weight = feature_weight
        self.TV_weight = TV_weight
        self.adversarial_weight = adversarial_weight

    def forward(self, HR_label, SR_label, HR, SR):
        pixel_loss = self.pixel_loss(SR, HR)
        feature_loss = self.feature_loss(SR, HR)
        smooth_loss = self.TV_loss(SR)
        adversarial_loss = self.adversarial_loss(SR_label, HR_label)
        generator_loss = self.pixel_weight * pixel_loss + self.feature_weight * feature_loss + self.TV_weight * smooth_loss + self.adversarial_weight * adversarial_loss
        return generator_loss


class VGG16_pretrained_feature_extractor(nn.Module):
    def __init__(self, judge_device):
        super(VGG16_pretrained_feature_extractor, self).__init__()
        pretrained_vgg = vgg16(weights='IMAGENET1K_V1').to(judge_device)
        self.block_0 = nn.Sequential(*list(pretrained_vgg.features)[0:5]).eval()
        self.block_1 = nn.Sequential(*list(pretrained_vgg.features)[5:10]).eval()
        self.block_2 = nn.Sequential(*list(pretrained_vgg.features)[10:17]).eval()
        self.block_3 = nn.Sequential(*list(pretrained_vgg.features)[17:24]).eval()
        self.block_4 = nn.Sequential(*list(pretrained_vgg.features)[24:31]).eval()
        for param in self.block_0.parameters():
            param.requires_grad = False
        for param in self.block_1.parameters():
            param.requires_grad = False
        for param in self.block_2.parameters():
            param.requires_grad = False
        for param in self.block_3.parameters():
            param.requires_grad = False
        for param in self.block_4.parameters():
            param.requires_grad = False

    def forward(self, SR, HR):
        SR_feature = self.block_0(SR)
        HR_feature = self.block_0(HR)
        loss_0 = F.mse_loss(SR_feature, HR_feature)
        SR_feature = self.block_1(SR_feature)
        HR_feature = self.block_1(HR_feature)
        loss_1 = F.mse_loss(SR_feature, HR_feature)
        SR_feature = self.block_2(SR_feature)
        HR_feature = self.block_2(HR_feature)
        loss_2 = F.mse_loss(SR_feature, HR_feature)
        SR_feature = self.block_3(SR_feature)
        HR_feature = self.block_3(HR_feature)
        loss_3 = F.mse_loss(SR_feature, HR_feature)
        SR_feature = self.block_4(SR_feature)
        HR_feature = self.block_4(HR_feature)
        loss_4 = F.mse_loss(SR_feature, HR_feature)
        return loss_0 + loss_1 + loss_2 + loss_3 + loss_4


class TV_loss(nn.Module):
    def __init__(self):
        super(TV_loss, self).__init__()

    def forward(self, X):
        batch_size = X.size()[0]
        count_h = X.size()[1] * (X.size()[2] - 1) * X.size()[3]
        count_w = X.size()[1] * X.size()[2] * (X.size()[3] - 1)
        h_tv = (torch.abs(X[:, :, 1:, :] - X[:, :, :-1, :])).sum()
        w_tv = (torch.abs(X[:, :, :, 1:] - X[:, :, :, :-1])).sum()
        return (h_tv / count_h + w_tv / count_w) / batch_size


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

first_time = False

HR_transform = transforms.Compose([transforms.ToPILImage(),
                                   transforms.CenterCrop(512),
                                   transforms.ToTensor()])
LR_transform = transforms.Compose([transforms.ToPILImage(),
                                   transforms.CenterCrop(512),
                                   transforms.Resize(512 // 4, interpolation=transforms.InterpolationMode.BICUBIC),
                                   transforms.ToTensor()])

train_set = RealSR.RealSRDataset('../data/RealSR/', True, True, 'V2', 'Nikon', LR_transform, HR_transform)
test_set = RealSR.RealSRDataset('../data/RealSR/', False, True, 'V2', 'Nikon', LR_transform, HR_transform)
train_data = DataLoader(train_set, batch_size=8, shuffle=True)
test_data = DataLoader(test_set, batch_size=8, shuffle=False)

for i in range(10):
    ax = plt.subplot(2, 5, i + 1)  # subplot: 可划分的画布
    idx = random.randint(0, len(train_set) - 1)
    digit_0 = train_set[idx][0].clone()  # 复制测试集tensor数据到digit_0
    digit_0 = digit_0.permute(1, 2, 0)  # 给matlib显示需要H W C
    digit_0_image = digit_0.reshape(128, 128, 3)  # 将打平的数据转换为image形式 128*128*3
    ax.imshow(digit_0_image, interpolation='nearest')  # 将内容加入画布
plt.show()  # 显示画布

for i in range(10):
    ax = plt.subplot(2, 5, i + 1)  # subplot: 可划分的画布
    idx = random.randint(0, len(train_set) - 1)
    digit_0 = train_set[idx][1].clone()  # 复制测试集tensor数据到digit_0
    digit_0 = digit_0.permute(1, 2, 0)  # 给matlib显示需要H W C
    digit_0_image = digit_0.reshape(512, 512, 3)  # 将打平的数据转换为image形式 128*128*3
    ax.imshow(digit_0_image, interpolation='nearest')  # 将内容加入画布
plt.show()  # 显示画布

generator = SRGAN.SRGAN_Generator(3).to(device)
discriminator = SRGAN.SRGAN_Discriminator(3).to(device)

criterion_generator = GeneratorLoss(judge_device=device)
criterion_discriminator = nn.BCELoss()
optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

if not first_time:
    generator.load_state_dict(torch.load('../saved_model/SRGAN/GNet.pth'))
    discriminator.load_state_dict(torch.load('../saved_model/SRGAN/DNet.pth'))
    generator.eval()
    discriminator.eval()


def train_loop(dataloader, train_device):
    generator.train()
    discriminator.train()
    size = len(dataloader.dataset)

    for batch, (LR, HR) in enumerate(dataloader):
        LR_dev = LR.to(train_device)
        HR_dev = HR.to(train_device)
        batch_size = LR_dev.shape[0]

        valid_label = torch.ones((batch_size, 1), requires_grad=False).to(train_device)
        fake_label = torch.zeros((batch_size, 1), requires_grad=False).to(train_device)

        optimizer_G.zero_grad()
        SR_dev = generator(LR_dev)
        SR_label_dev = discriminator(SR_dev)
        HR_label_dev = valid_label
        G_loss = criterion_generator(HR_label_dev, SR_label_dev, HR_dev, SR_dev)
        G_loss.backward()
        optimizer_G.step()

        optimizer_D.zero_grad()
        D_HR_loss = criterion_discriminator(discriminator(HR_dev), valid_label)
        D_SR_loss = criterion_discriminator(discriminator(SR_dev.detach()), fake_label)
        D_loss = D_HR_loss + D_SR_loss
        D_loss.backward()
        optimizer_D.step()

        if batch % 100 == 0:
            g_loss, d_loss, current = G_loss.item(), D_loss.item(), batch * batch_size
            print(f"g_loss: {g_loss:>7f} d_loss: {d_loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, test_device):
    generator.eval()
    discriminator.eval()
    for batch, (LR, HR) in enumerate(dataloader):
        if batch == 2:
            LR_test = LR[0:3]
            HR_test = HR[0:3]
            LR_test = LR_test.to(test_device)
            HR_test = HR_test.to(test_device)
            SR_test = generator(LR_test)
            for j in range(3):
                original = LR_test.permute(0, 2, 3, 1).detach().cpu().numpy()[j]
                '''
                origin_min = np.min(original, axis=(0, 1), keepdims=True)
                origin_max = np.max(original, axis=(0, 1), keepdims=True)
                original = (original - origin_min) / (origin_max - origin_min)
                '''
                original = original.clip(0, 1)
                plot_ax_1 = plt.subplot(1, 3, 1)
                plot_ax_1.imshow(original, interpolation='nearest')
                recon = SR_test.permute(0, 2, 3, 1).detach().cpu().numpy()[j]
                '''
                recon_min = np.min(recon, axis=(0, 1), keepdims=True)
                recon_max = np.max(recon, axis=(0, 1), keepdims=True)
                recon = (recon - recon_min) / (recon_max - recon_min)
                '''
                recon = recon.clip(0, 1)
                plot_ax_2 = plt.subplot(1, 3, 2)
                plot_ax_2.imshow(recon, interpolation='nearest')
                hr = HR_test.permute(0, 2, 3, 1).detach().cpu().numpy()[j]
                '''
                hr_min = np.min(hr, axis=(0, 1), keepdims=True)
                hr_max = np.max(hr, axis=(0, 1), keepdims=True)
                hr = (hr - hr_min) / (hr_max - hr_min)
                '''
                hr = hr.clip(0, 1)
                plot_ax_3 = plt.subplot(1, 3, 3)
                plot_ax_3.imshow(hr, interpolation='nearest')
                plt.show()


epochs = 100
for train_num in range(epochs):
    print(f"Epoch {train_num + 1}\n-------------------------------")
    train_loop(train_data, device)
    torch.save(generator.state_dict(), '../saved_model/SRGAN/GNet.pth')
    torch.save(discriminator.state_dict(), '../saved_model/SRGAN/DNet.pth')
    # test_loop(test_data, device)
print("Done!")

test_loop(test_data, device)
