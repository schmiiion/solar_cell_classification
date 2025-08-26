from torch.utils.data import Dataset
import torch
from pathlib import Path
from skimage.io import imread
from skimage.color import gray2rgb
import numpy as np
import torchvision as tv

train_mean = [0.59685254, 0.59685254, 0.59685254]
train_std = [0.16043035, 0.16043035, 0.16043035]


class ChallengeDataset(Dataset):

    def __init__(self, data, mode: str = 'train'):
        """Transform must atleast contain ToPILImage(), ToTensor(), Normalize()"""
        self.data = data
        self.mode = mode

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_info = self.data.iloc[idx][0]
        img_path, crack, defect = img_info.split(';')
        img = imread(img_path)
        img = gray2rgb(img)

        if self.mode == 'train':
            img_tensor = self._train_transform(img)
        else:
            img_tensor = self._val_transform(img)
        return img_tensor, torch.tensor([int(crack), int(defect)], dtype=torch.float32)

    def _train_transform(self, img):
        transform = tv.transforms.Compose([
            tv.transforms.ToPILImage(),
            tv.transforms.RandomHorizontalFlip(),
            tv.transforms.RandomVerticalFlip(),
            tv.transforms.RandomRotation(15),
            tv.transforms.ColorJitter(brightness=0.2, contrast=0.2),
            tv.transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
            tv.transforms.GaussianBlur(3, sigma=(0.1, 1.5)),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(train_mean, train_std)
        ])
        return transform(img)

    def _val_transform(self, img):
        transform = tv.transforms.Compose([
            tv.transforms.ToPILImage(),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(train_mean, train_std)
        ])
        return transform(img)
