import os
from torch.utils.data.dataset import Dataset
from torchvision.io import read_image


class RealSRDataset(Dataset):
    def __init__(self, main_dir, is_train, is_auto_LR, dataset_version, camera_version, LR_transform, HR_transform):
        super(RealSRDataset, self).__init__()
        self.main_dir = main_dir
        self.is_train = is_train
        self.is_auto_LR = is_auto_LR
        self.dataset_version = dataset_version
        self.camera_version = camera_version
        self.LR_transform = LR_transform
        self.HR_transform = HR_transform

    def __getitem__(self, index):
        if self.is_train is True:
            img_path = os.path.join(self.main_dir, 'RealSR_V2/Nikon/train/x4/HR/'+'Nikon_'+str(index+1).zfill(3)+'.png')
        else :
            img_path = os.path.join(self.main_dir, 'RealSR_V2/Nikon/test/x4/HR/'+'Nikon_'+str(index+1).zfill(3)+'.png')
        image = read_image(img_path)
        opt_LR = self.LR_transform(image)
        opt_HR = self.HR_transform(image)
        return opt_LR, opt_HR

    def __len__(self):
        if self.is_train is True:
            length = 200
        else:
            length = 50
        return length



