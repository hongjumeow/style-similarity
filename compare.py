import os
import argparse
import math
from tqdm import tqdm

import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

from VGGnet import VGG
from style_distance import calc_style_distance

device = torch.device("cuda" if (torch.cuda.is_available()) else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('--dataset1', type=str)
parser.add_argument('--dataset2', type=str)

def image_loader(path):
    image = Image.open(path)

    loader = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])

    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)

def is_image_file(name):
    if name.endswith('.jpg') or name.endswith('.png'):
        return True
    else:
        return False

def file_loader(path):
    imgs = []
    for root, dirs, fnames in sorted(os.walk(path, followlinks=True)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                imgs.append(path)
    return imgs

class Compare():
    def __init__(self, dataset1, dataset2):
        self.freq = 100

        self.dataset1 = file_loader(dataset1)
        self.dataset2 = file_loader(dataset2)

        self.model = VGG()
        with torch.no_grad():
            self.model.to(device).eval()

        self.feats1, self.feats2 = self._get_features()

    def _get_features(self):
        length = len(self.dataset1)
        if len(self.dataset1) > len(self.dataset2):
            length = len(self.dataset2)

        feats1 = [0] * 5
        feats2 = [0] * 5
        for i in tqdm(range(length)):
            with torch.no_grad():
                img1 = image_loader(self.dataset1[i])
                img2 = image_loader(self.dataset2[i])
                img1_feats = self.model(img1)
                img2_feats = self.model(img2)

                if i == 0:
                    feats1 = [a for a in img1_feats]
                    feats2 = [a for a in img2_feats]
                else:
                    feats1 = [a + b for a, b in zip(feats1, img1_feats)]
                    feats2 = [a + b for a, b in zip(feats2, img2_feats)]
        
        feats1_mean = [a / length for a in feats1]
        feats2_mean = [a / length for a in feats2]
        return feats1_mean, feats2_mean

    def get_distance(self):
        distance = 0
        for feat1, feat2 in zip(self.feats1, self.feats2):
            distance += calc_style_distance(feat1, feat2)
        
        distance_neat = math.sqrt(distance.item())
        
        return distance_neat

if __name__ == "__main__":
    opt = parser.parse_args()

    compare = Compare(opt.dataset1, opt.dataset2)
    print(compare.get_distance())