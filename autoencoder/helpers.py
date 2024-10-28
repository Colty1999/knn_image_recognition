from torchvision import transforms
import torch
from torch import nn
import torch.nn.functional as F

reconstruction_function = nn.BCELoss()
reconstruction_function.reduction = 'sum'


def loss_function(reconstructed, prototypes, mu, logvar):
    # Reconstruction loss
    BCE = reconstruction_function(reconstructed, prototypes)
    
    # KL Divergence loss
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element).mul_(-0.5)
    
    return BCE + KLD


train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
])

prototype_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])
