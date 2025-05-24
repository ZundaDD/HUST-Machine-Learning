from torchvision.models import resnet18,resnet34,vgg11_bn,resnet50
from torchvision import transforms as T
import torch.nn as nn


def get_model(model_key, pretrained=False):
    if model_key == 'CNN':
        model = CNN()
        reshape = T.Compose([])

    elif model_key == 'resnet34':
        model = resnet34(pretrained=pretrained)
        reshape = T.Compose([])
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model.fc = nn.Linear(512, 10)

    elif model_key == 'resnet50':
        model = resnet50(pretrained=pretrained)
        reshape = T.Compose([])
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model.fc = nn.Linear(2048, 10)

    elif model_key == 'vgg11_bn':
        model = vgg11_bn(pretrained=pretrained)
        reshape = T.Compose([T.Resize((32, 32))])
        model.features[0] = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.classifier[-1] = nn.Linear(4096, 10)

    else:
        model = resnet18(pretrained=pretrained)
        reshape = T.Compose([])
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model.fc = nn.Linear(512, 10)

    return model, reshape

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=5, stride=1, padding=0)
        self.maxpool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=5, stride=1, padding=0)
        self.maxpool2 = nn.MaxPool2d(2)
        self.linear1 = nn.Linear(320, 128)
        self.linear2 = nn.Linear(128, 64)
        self.linear3 = nn.Linear(64, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.maxpool1(self.conv1(x)))
        x = self.relu(self.maxpool2(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)

        return x