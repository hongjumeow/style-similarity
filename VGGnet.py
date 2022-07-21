import torch
import torch.nn as nn
import torchvision.models as models
from tqdm import tqdm

from loader import image_loader

class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.req_features = ['0', '5', '10', '19', '28']

        self.model = models.vgg19(pretrained=True).features[:29]
        # model containing the first 29 layers

    def forward(self, x):
        features = []

        for layer_num, layer in enumerate(self.model):
            x = layer(x)
            if(str(layer_num) in self.req_features):
                features.append(x)

        return features
    
    def get_features(self, dataset1, dataset2, **args):
        num_images = args['num_images']
        
        feats1 = [0] * 5
        feats2 = [0] * 5
        for i in tqdm(range(num_images)):
            with torch.no_grad():
                img1_feats = self.forward(image_loader(dataset1[i]).to(args['device'], torch.float))
                img2_feats = self.forward(image_loader(dataset2[i]).to(args['device'], torch.float))

                if i == 0:
                    feats1 = [a for a in img1_feats]
                    feats2 = [a for a in img2_feats]
                else:
                    feats1 = [a + b for a, b in zip(feats1, img1_feats)]
                    feats2 = [a + b for a, b in zip(feats2, img2_feats)]
        
        feats1_mean = [a / num_images for a in feats1]
        feats2_mean = [a / num_images for a in feats2]

        return feats1_mean, feats2_mean