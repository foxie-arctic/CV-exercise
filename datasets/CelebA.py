import os
from torch.utils.data.dataset import Dataset
from torchvision.io import read_image


class CelebADataset(Dataset):
    def __init__(self, img_dir, is_train=True, transform=None, target_transform=None):
        super(CelebADataset, self).__init__()
        self.img_dir = img_dir
        self.is_train = is_train
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        if self.is_train is True:
            required_str = str(index + 1).zfill(6)
            required_dir = required_str + '.jpg'
        else:
            required_str = str(index + 200001).zfill(6)
            required_dir = required_str + '.jpg'

        required_img_dir = os.path.join(self.img_dir, required_dir)
        image = read_image(required_img_dir)
        if self.transform is None:
            opt_image = image
        else:
            opt_image = self.transform(image)
        label = 0
        if self.target_transform is None:
            opt_label = label
        else:
            opt_label = self.target_transform(label)
        return opt_image, opt_label

    def __len__(self):
        if self.is_train is True:
            count = 200000
        else:
            count = 2599
        return count


'''   
image_transform = transforms.Compose([transforms.ToPILImage(),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.CenterCrop(148),
                                      transforms.Resize([64, 64]),
                                      transforms.ToTensor()])
train_set = CelebADataset('./data/img_align_celeba/img_align_celeba/', is_train=True, transform=image_transform,
                          target_transform=None)
test_set = CelebADataset('./data/img_align_celeba/img_align_celeba/', is_train=False, transform=image_transform,
                         target_transform=None)
train_data = DataLoader(train_set, batch_size=64, shuffle=True)
test_data = DataLoader(test_set, batch_size=64, shuffle=False)

for i in range(10):  # [0,3]
    ax = plt.subplot(2, 5, i + 1)  # subplot: 可划分的画布
    idx = random.randint(0, len(train_set))
    digit_0 = train_set[idx][0].clone()  # 复制测试集tensor数据到digit_0
    digit_0 = digit_0.permute(1, 2, 0)  # 给matlib显示需要H W C
    digit_0_image = digit_0.reshape(64, 64, 3)  # 将打平的数据转换为image形式 128*128*3
    ax.imshow(digit_0_image, interpolation='nearest')  # 将内容加入画布
    ax.set_title('label: {}'.format(train_set[idx][1]), fontsize=10, color='black')  # 标题为：label: label string/num
plt.show()  # 显示画布
'''
