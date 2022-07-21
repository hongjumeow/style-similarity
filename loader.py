import os
import torch
import torchvision.transforms as transforms
from PIL import Image

def image_loader(path, **args):
    image = Image.open(path)

    loader = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])

    image = loader(image).unsqueeze(0)
    return image

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