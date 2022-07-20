import torch

def calc_style_distance(feat1, feat2):
    batch_size, channel, height, width = feat1.shape

    feat1_shaped = feat1.view(channel, height * width)
    feat2_shaped = feat2.view(channel, height * width)

    G = torch.mm(feat1_shaped, feat1_shaped.t())
    A = torch.mm(feat2_shaped, feat2_shaped.t())

    distance = torch.mean((G-A) ** 2)
    return distance
