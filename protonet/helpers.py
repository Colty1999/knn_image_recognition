import torch
from torch import nn
from PIL import ImageOps, Image
import random
from torchvision import transforms

# Define individual augmentation functions
class RandomCrop:
    def __init__(self, size, padding=0):
        self.size = size
        self.padding = padding

    def __call__(self, img):
        if self.padding > 0:
            img = ImageOps.expand(img, border=self.padding, fill=0)
        w, h = img.size
        th, tw = self.size, self.size
        if w == tw and h == th:
            return img
        if w < tw or h < th:
            return img.resize((tw, th), Image.BILINEAR)
        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        return img.crop((x1, y1, x1 + tw, y1 + th))

class RandomHorizontallyFlip:
    def __call__(self, img):
        if random.random() < 0.5:
            return img.transpose(Image.FLIP_LEFT_RIGHT)
        return img

class RandomRotate:
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, img):
        rotate_degree = random.uniform(-self.degree, self.degree)
        return img.rotate(rotate_degree, Image.BILINEAR)

class Scale:
    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        w, h = img.size
        if w < h:
            ow = self.size
            oh = int(self.size * h / w)
            return img.resize((ow, oh), Image.BILINEAR)
        else:
            oh = self.size
            ow = int(self.size * w / h)
            return img.resize((ow, oh), Image.BILINEAR)

class CenterCrop:
    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        w, h = img.size
        th, tw = self.size, self.size
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))
        return img.crop((x1, y1, x1 + tw, y1 + th))

# Define the composed transforms
train_transform = transforms.Compose([
    Scale(256),
    RandomCrop(224),
    RandomHorizontallyFlip(),
    RandomRotate(15)
])

prototype_transform = transforms.Compose([
    Scale(256),
    CenterCrop(224)
])


reconstruction_function = nn.BCELoss(reduction='sum')

def loss_function(reconstructed, prototypes, mu, logvar, beta=0.1):
    BCE = reconstruction_function(reconstructed, prototypes)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / mu.size(0)
    return BCE + beta * KLD

