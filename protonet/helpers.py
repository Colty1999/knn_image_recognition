import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.data.sampler import Sampler
from sklearn.model_selection import StratifiedShuffleSplit


class PrototypicalBatchSampler(Sampler):
    """
    PrototypicalBatchSampler generates indices for a fixed number of classes per iteration,
    with a fixed number of support and query samples per class.
    """

    def __init__(self, labels, classes_per_it, num_samples, iterations):
        """
        Args:
        - labels: array-like, labels for the dataset
        - classes_per_it: number of random classes per iteration (N-way)
        - num_samples: number of samples per class (support + query)
        - iterations: number of iterations per epoch
        """
        self.labels = labels
        self.classes_per_it = classes_per_it
        self.num_samples = num_samples
        self.iterations = iterations
        self.classes = np.unique(self.labels)

    def __iter__(self):
        for _ in range(self.iterations):
            batch = []
            chosen_classes = np.random.choice(self.classes, self.classes_per_it, replace=False)
            for class_ in chosen_classes:
                class_indices = np.where(self.labels == class_)[0]
                # print(f"Class {class_} has {len(class_indices)} samples available.")
                selected_indices = np.random.choice(class_indices, self.num_samples, replace=False)
                batch.extend(selected_indices)
            yield batch

    def __len__(self):
        return self.iterations


def load_data(data_dir, validation_split=0.2, classes_per_it=5, num_samples=10, iterations=100):
    """
    Load data for Prototypical Networks with custom batch sampling.
    Args:
    - data_dir: Directory where the dataset is located
    - validation_split: Fraction of dataset to use as validation
    - classes_per_it: Number of classes per iteration (N-way)
    - num_samples: Number of support + query samples per class
    - iterations: Number of iterations per epoch
    """
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Mean and std from ImageNet
        ]),
        'val': transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }


    dataset = datasets.ImageFolder(data_dir, transform=data_transforms['train'])
    dataset_size = len(dataset)
    # indices = list(range(dataset_size))
    # split = int(validation_split * dataset_size)
    # random.shuffle(indices)
    # train_indices, val_indices = indices[split:], indices[:split]
    
    # Get the targets (labels) from the dataset
    labels = dataset.targets
    # Use StratifiedShuffleSplit to split the dataset such that each class is equally represented
    stratified_split = StratifiedShuffleSplit(n_splits=1, test_size=validation_split)
    train_indices, val_indices = next(stratified_split.split(np.zeros(len(labels)), labels))


    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)

    # Get the labels from the dataset (assuming your dataset has a .targets or .labels attribute)
    train_labels = [train_dataset.dataset.targets[idx] for idx in train_indices]
    val_labels = [val_dataset.dataset.targets[idx] for idx in val_indices]
    print(f"Total training samples: {len(train_indices)}")
    print(f"Total validation samples: {len(val_indices)}")
    # from collections import Counter
    # print(Counter(train_labels)) 

    # Prototypical Batch Sampler for Few-Shot Learning
    train_sampler = PrototypicalBatchSampler(labels=train_labels, classes_per_it=classes_per_it, num_samples=num_samples, iterations=iterations)
    val_sampler = PrototypicalBatchSampler(labels=val_labels, classes_per_it=classes_per_it, num_samples=num_samples, iterations=iterations)

    dataloaders = {
        'train': DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=4),
        'val': DataLoader(val_dataset, batch_sampler=val_sampler, num_workers=4)
    }

    dataset_sizes = {
        'train': len(train_indices),
        'val': len(val_indices)
    }
    class_names = dataset.classes

    return dataloaders, dataset_sizes, class_names
