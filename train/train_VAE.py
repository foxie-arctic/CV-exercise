import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torch import optim

import matplotlib.pyplot as plt
import random

from datasets import CelebA
from models import VanillaVAE
from tqdm import *
import time

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

image_transform = transforms.Compose([transforms.ToPILImage(),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.CenterCrop(148),
                                      transforms.Resize([64, 64]),
                                      transforms.ToTensor()])
train_set = CelebA.CelebADataset('../data/img_align_celeba/img_align_celeba/', is_train=True, transform=image_transform,
                                 target_transform=None)
test_set = CelebA.CelebADataset('../data/img_align_celeba/img_align_celeba/', is_train=False, transform=image_transform,
                                target_transform=None)
train_data = DataLoader(train_set, batch_size=64, shuffle=True)
test_data = DataLoader(test_set, batch_size=64, shuffle=False)

for i in range(10):  # [0,3]
    ax = plt.subplot(2, 5, i + 1)  # subplot: 可划分的画布
    idx = random.randint(0, len(train_set)-1)
    digit_0 = train_set[idx][0].clone()  # 复制测试集tensor数据到digit_0
    digit_0 = digit_0.permute(1, 2, 0)  # 给matlib显示需要H W C
    print(digit_0)
    digit_0_image = digit_0.reshape(64, 64, 3)  # 将打平的数据转换为image形式 128*128*3
    ax.imshow(digit_0_image, interpolation='nearest')  # 将内容加入画布
    ax.set_title('label: {}'.format(train_set[idx][1]), fontsize=10, color='black')  # 标题为：label: label string/num
plt.show()  # 显示画布

model_VAE = VanillaVAE.VariantAutoEncoder().to(device)
print(model_VAE)
criterion_VAE = model_VAE.loss_function
optimizer_VAE = optim.SGD(model_VAE.parameters(), lr=1e-3, momentum=0.9, weight_decay=0)


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        X_dev = X.to(device)
        result_args = model(X_dev)
        loss = loss_fn(*result_args)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(model):
    for j in range(5):
        plot_ax = plt.subplot(1, 5, j + 1)
        digit_sampled = model.sample(1, device)
        digit_sampled = digit_sampled.cpu()
        digit_sampled = digit_sampled.detach()
        digit_sampled = digit_sampled.reshape(3, 64, 64)
        digit_sampled_matlib = digit_sampled.permute(1, 2, 0)
        digit_image = digit_sampled_matlib.reshape(64, 64, 3)
        plot_ax.imshow(digit_image, interpolation='nearest')
    plt.show()


epochs = 20
for t in range(epochs):
    print(f"Epoch {t + 1}\n-------------------------------")
    train_loop(train_data, model_VAE, criterion_VAE, optimizer_VAE)
    if t % 10 == 0:
        test_loop(model_VAE)
print("Done!")
test_loop(model_VAE)
