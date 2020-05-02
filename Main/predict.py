import torch
from torch import nn

from tqdm import tqdm
import numpy as np
import matplotlib.pylab as plt
import torchvision.models as tvmodels
from torchvision import datasets, transforms
import dataset
from modelv2 import CaptchaNN
import os
from PIL import Image
from torch.utils.data import DataLoader


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = "captcha-breaker-v%d.pth" % CaptchaNN.version()
    data_path = "./predict"

    net = CaptchaNN()
    net = net.to(device)
    net.load_state_dict(torch.load(model_path))

    #transform = dataset.CaptchaDataset.get_transform(224, 224)

    train_dataset = dataset.CaptchaDataset(data_path, 224, 224)
    trainIter = DataLoader(train_dataset, batch_size=1, num_workers=0,
                                   shuffle=False, drop_last=False)
    for i, (X, label) in enumerate(trainIter):
        X = X.to(device)
        label = label.to(device)
        label = label.long()

        label1 = label[:, 0]
        label2 = label[:, 1]
        label3 = label[:, 2]
        label4 = label[:, 3]

        y1, y2, y3, y4 = net(X)

        _, y1_pred = torch.max(y1.data, dim=1)
        _, y2_pred = torch.max(y2.data, dim=1)
        _, y3_pred = torch.max(y3.data, dim=1)
        _, y4_pred = torch.max(y4.data, dim=1)

        print(bool(y1_pred == label1), bool(y2_pred == label2), bool(y3_pred == label3), bool(y4_pred == label4))



if __name__ == '__main__':
    main()
