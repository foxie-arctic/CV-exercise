from torch.utils.data.dataset import Dataset
import numpy as np
import scipy.io as sio
from PIL import Image


class SVHNDataset(Dataset):
    def __init__(self, mat_dir, transform=None, target_transform=None):
        super(SVHNDataset, self).__init__()
        self.raw_mat = sio.loadmat(mat_dir)
        self.labels = self.raw_mat['y']
        self.images = self.raw_mat['X']
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        image_arr = self.images[:, :, :, index]
        label = self.labels[index, 0]
        if self.transform is None:
            t_image = image_arr
        else:
            image = Image.fromarray(image_arr, 'RGB')
            t_image = self.transform(image)
        if label == 10:
            label = 0
        if self.target_transform is None:
            t_label = label
        else:
            t_label = self.target_transform(label)
        return t_image, t_label

    def __len__(self):
        count = np.size(self.labels)
        return count


'''
image_transform = transforms.Compose(
    [transforms.RandomRotation(15),
     transforms.ToTensor()])  # H W C---C H W
train_set = SVHNDataset('./data/SVHN/train_32x32.mat', transform=image_transform, target_transform=None)
test_set = SVHNDataset('./data/SVHN/test_32x32.mat', transform=image_transform, target_transform=None)
train_data = DataLoader(train_set, batch_size=128, shuffle=True)
test_data = DataLoader(test_set, batch_size=256, shuffle=False)
'''
