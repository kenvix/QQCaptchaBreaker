import os
from PIL import Image
from torch.utils import data
import numpy as np
from torchvision import transforms
import torch
from torch.utils.data import DataLoader
from sklearn import preprocessing


def getCaptchaDataset(batch_size, root_path, width=168, height=64):
    train_dataset = CaptchaDataset(root_path + '/train', width, height)
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=0,
                                   shuffle=True, drop_last=True)
    test_data = CaptchaDataset(root_path + '/test', width, height)
    test_data_loader = DataLoader(test_data, batch_size=batch_size,
                                  num_workers=0, shuffle=True, drop_last=True)
    return train_data_loader, test_data_loader


class CaptchaDataset(torch.utils.data.Dataset):
    def __init__(self, dir_path, width=168, height=64):
        super(CaptchaDataset, self).__init__()
        self.width = width
        self.height = height

        self.images_paths = [os.path.join(dir_path, img) for img in os.listdir(dir_path)]
        self.transform = transforms.Compose([
            transforms.Resize((self.height, self.width)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return len(self.images_paths)

    @staticmethod
    def to_label(string):
        label = []
        for i in range(len(string)):
            label.append(string[i])

        le = preprocessing.LabelEncoder()
        targets = le.fit_transform(label)
        return targets

    def __getitem__(self, index):
        img = self.images_paths[index]
        label = img.replace('\\', '/').split("/")[-1].split(".")[0]
        asLabel = self.to_label(label)
        labelTensor = torch.Tensor(asLabel)
        datao = Image.open(img)
        data = self.transform(datao)
        return data, labelTensor


def main():

    pass


if __name__ == '__main__':
    main()
