import torch
from torch import nn

from tqdm import tqdm
import numpy as np
import matplotlib.pylab as plt
import torchvision.models as tvmodels
from torchvision import datasets, transforms
import dataset


class CaptchaNN(nn.Module):
    @staticmethod
    def version():
        return 1

    def __init__(self):
        super(CaptchaNN, self).__init__()
        self.kernel_size = 5
        self.class_num = 26

        self.net = nn.Sequential( #168x64
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=self.kernel_size, padding=(1, 1)),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(6),
            nn.ReLU(),

            nn.Conv2d(in_channels=6, out_channels=12, kernel_size=self.kernel_size, padding=(1, 1)),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(12),
            nn.ReLU(),

            nn.Conv2d(in_channels=12, out_channels=24, kernel_size=self.kernel_size, padding=(1, 1)),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(24),
            nn.ReLU(),
        )
        self.linear = nn.Sequential(
            nn.Linear(24*6*19, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )

        self.fc1 = nn.Linear(128, self.class_num)
        self.fc2 = nn.Linear(128, self.class_num)
        self.fc3 = nn.Linear(128, self.class_num)
        self.fc4 = nn.Linear(128, self.class_num)

    def forward(self, x):
        x = self.net(x)
        x = x.view(-1, 24*6*19)
        x = self.linear(x)
        x1 = self.fc1(x)
        x2 = self.fc2(x)
        x3 = self.fc3(x)
        x4 = self.fc4(x)
        return x1, x2, x3, x4

