import urllib
import os

from torch.autograd import Variable

from dataset import CaptchaDataset
from modelv2 import CaptchaNN
import os
import torch
from PIL import Image


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = "captcha-breaker-v%d.pth" % CaptchaNN.version()
    data_path = "./fetched"
    getImageUrl = "http://captcha.qq.com/getimage"
    downloadNum = 500

    net = CaptchaNN()
    net = net.to(device)
    net.load_state_dict(torch.load(model_path))
    net.eval()

    transform = CaptchaDataset.get_transform(224, 224)

    for i in range(0, downloadNum):
        file = "!unclassified.jpg"
        url = getImageUrl
        localPath = os.path.join(data_path, file)
        urllib.request.urlretrieve(url, localPath)

        pilImg = Image.open(localPath)
        img = transform(pilImg)
        img = CaptchaDataset.to_var(img)
        X = img.to(device)
        pred = CaptchaDataset.decode_label(net.predict(X))
        os.rename(localPath, os.path.join(data_path, pred + ".jpg"))

        print("Downloaded and recognized as ", pred)
    pass


if __name__ == '__main__':
    main()