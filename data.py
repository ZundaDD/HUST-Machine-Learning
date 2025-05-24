import os
from torch.utils.data.dataset import Dataset
from torchvision.models import resnet18,resnet34,vgg11_bn,resnet50
from torchvision import transforms as T
from PIL import Image

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
        img_item_path = os.path.join(self.root_dir, self.label[idx], img_name)
        img = Image.open(img_item_path)
        img = T.ToTensor()(img)
        return img, int(self.label[idx]), img_name

    def __len__(self):
        return len(self.img_path)
