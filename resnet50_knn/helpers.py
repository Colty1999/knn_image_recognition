import random
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


def load_data(data_dir, batch_size=32, validation_split=0.2):
    dataset = datasets.ImageFolder(data_dir, transform=data_transforms['train'])
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(validation_split * dataset_size)
    random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)

    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4),
        'val': DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    }
    dataset_sizes = {
        'train': len(train_indices),
        'val': len(val_indices)
    }
    class_names = dataset.classes

    return dataloaders, dataset_sizes, class_names


def extract_features(model, dataloader):
    model.eval()
    features = []
    labels = []

    with torch.no_grad():
        for inputs, lbls in dataloader:
            inputs = inputs.to(device)
            lbls = lbls.to(device)

            outputs = model(inputs)
            features.append(outputs.cpu().numpy())
            labels.append(lbls.cpu().numpy())

    features = np.vstack(features)
    labels = np.hstack(labels)

    return features, labels
