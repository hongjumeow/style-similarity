import os
import argparse
import math
import torch

from VGGnet import VGG
from loader import file_loader

device = torch.device("cuda" if (torch.cuda.is_available()) else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('--dataset1', type=str)
parser.add_argument('--dataset2', type=str)
parser.add_argument('--num_images', type=int, default=None)

class Compare():
    def __init__(self, dataset1, dataset2, num_images):
        self.freq = 100

        self.dataset1 = file_loader(dataset1)
        self.dataset2 = file_loader(dataset2)

        self.num_images = num_images

        self.model = VGG()
        with torch.no_grad():
            self.model.to(device).eval()

        self.feats1, self.feats2 = self._get_features()

    def _get_features(self):
         
        num_images = len(self.dataset1)
        if len(self.dataset1) > len(self.dataset2):
            num_images = len(self.dataset2)
        if self.num_images != None:
            num_images = self.num_images

        return self.model.get_features(self.dataset1, self.dataset2, num_images=num_images, device=device)

    def calc_style_distance(self):
        distances = 0
        for feat1, feat2 in zip(self.feats1, self.feats2):
            batch_size, channel, height, width = feat1.shape

            feat1_shaped = feat1.view(channel, height * width)
            feat2_shaped = feat2.view(channel, height * width)

            G = torch.mm(feat1_shaped, feat1_shaped.t())
            A = torch.mm(feat2_shaped, feat2_shaped.t())

            distance = torch.mean((G-A) ** 2)
            distances += distance
        
        distance_neat = math.sqrt(distances.item())
        
        return distance_neat

if __name__ == "__main__":
    opt = parser.parse_args()

    compare = Compare(opt.dataset1, opt.dataset2, opt.num_images)
    print(compare.calc_style_distance())