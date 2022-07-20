import torch
import torch.nn as nn
import torchvision.models as models

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