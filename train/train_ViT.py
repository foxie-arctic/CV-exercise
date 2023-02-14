import os
import torch
import torch.nn.functional as F
import torchvision
from torch import tensor
import numpy as np
import pandas as pd
from torch.utils.data.dataset import Dataset
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.io import read_image
import skimage.io as skimio
import scipy.io as sio

import torch.nn as nn
from torch import optim

import matplotlib.pyplot as plt
import random

from datasets import SVHN
from models import VanillaViT

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

image_transform = transforms.Compose(
    [transforms.RandomRotation(15),
     transforms.ToTensor()])  # H W C---C H W
train_set = SVHN.SVHNDataset('../data/SVHN/train_32x32.mat', transform=image_transform, target_transform=None)
test_set = SVHN.SVHNDataset('../data/SVHN/test_32x32.mat', transform=image_transform, target_transform=None)
train_data = DataLoader(train_set, batch_size=256, shuffle=True)
test_data = DataLoader(test_set, batch_size=256, shuffle=False)

for i in range(10):  # [0,3]
    ax = plt.subplot(2, 5, i + 1)  # subplot: 可划分的画布
    idx = random.randint(0, len(train_set)-1)
    digit_0 = train_set[idx][0].clone()  # 复制测试集tensor数据到digit_0
    digit_0 = digit_0.permute(1, 2, 0)  # 给matlib显示需要H W C
    digit_0_image = digit_0.reshape(32, 32, 3)  # 将打平的数据转换为image形式 32*32*3
    ax.imshow(digit_0_image, interpolation='nearest')  # 将内容加入画布
    ax.set_title('label: {}'.format(train_set[idx][1]), fontsize=10, color='black')  # 标题为：label: label string/num
plt.show()  # 显示画布

model_ViT = VanillaViT.VanillaViT().to(device)
print(model_ViT)
criterion_ViT = nn.CrossEntropyLoss()
optimizer_ViT = optim.Adam(model_ViT.parameters(), lr=1e-4)

'''
for batch, (X,y) in enumerate(test_data):
    if batch == 0:
        y_oh=F.one_hot(y,num_classes=10)
        print(y_oh.shape)
'''


def train_loop(dataloader, model, loss_fn, optimizer, train_device):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X_dev = X.to(train_device)
        '''
        one_hot_y = F.one_hot(y, num_classes=10)
        one_hot_y_dev = one_hot_y.to(train_device)
        '''
        y_dev = y.to(train_device)
        res = model(X_dev)
        loss = loss_fn(res, y_dev)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn, test_device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    total_loss = 0
    correct = 0
    model.eval()
    with torch.no_grad():
        for batch, (X, y) in enumerate(dataloader):
            X_dev = X.to(test_device)
            '''
            one_hot_y = F.one_hot(y, num_classes=10)
            one_hot_y_dev = one_hot_y.to(test_device)
            '''
            y_dev = y.to(test_device)
            res = model(X_dev)
            loss = loss_fn(res, y_dev)
            total_loss += loss
            correct += (res.argmax(1) == y_dev).type(torch.float).sum().item()
    test_loss = total_loss / num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


epochs = 20
for t in range(epochs):
    print(f"Epoch {t + 1}\n-------------------------------")
    train_loop(train_data, model_ViT, criterion_ViT, optimizer_ViT, device)
    '''
    if t % 10 == 0:
        test_loop(test_data, model_ViT, criterion_ViT, device)
    '''

print("Done!")
test_loop(test_data, model_ViT, criterion_ViT, device)
