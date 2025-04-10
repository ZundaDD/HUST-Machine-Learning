import os
import torch
from torch.utils.data.dataset import Dataset, Subset
from glob import glob
import torch.nn as nn
from torchvision.models import resnet18,resnet34,vgg11_bn,resnet50
from torchvision import transforms as T
from PIL import Image
import pandas as pd

class EasyDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        label_dir = os.listdir(self.root_dir)
        self.path = [os.path.join(self.root_dir, label) for label in label_dir]
        self.label = []
        self.img_path = []
        for path in label_dir:
            imgs = os.listdir(os.path.join(self.root_dir, path))
            self.label.extend(path * len(imgs))

            self.img_path.extend(imgs)

    def __getitem__(self, idx):
        img_name = self.img_path[idx]
        img_item_path = os.path.join(self.root_dir, self.label[idx], img_name)  # 与文件的路径拼接起来
        img = Image.open(img_item_path)
        img = T.ToTensor()(img)
        return img, int(self.label[idx]), img_name

    def __len__(self):
        return len(self.img_path)


def get_model(model_key):
    if model_key == 'resnet34':
        model = resnet34(pretrained=True)
        reshape = T.Compose([])
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model.fc = nn.Linear(512, 10)

    elif model_key == 'resnet50':
        model = resnet50(pretrained=True)
        reshape = T.Compose([])
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model.fc = nn.Linear(2048, 10)

    elif model_key == 'vgg11_bn':
        model = vgg11_bn(pretrained=True)
        reshape = T.Compose([
            T.Resize((32, 32))
        ])
        model.features[0] = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.classifier[-1] = nn.Linear(4096, 10)

    else:
        model = resnet18(pretrained=True)
        reshape = T.Compose([])
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model.fc = nn.Linear(512, 10)

    return model, reshape