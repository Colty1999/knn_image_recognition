import random
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import numpy as np
import csv

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
    image_paths = []

    # Access the underlying dataset
    base_dataset = dataloader.dataset.dataset  # Access the original ImageFolder dataset
    subset_indices = dataloader.dataset.indices  # Get the indices of the Subset

    with torch.no_grad():
        for inputs, lbls in dataloader:
            inputs = inputs.to(device)
            lbls = lbls.to(device)

            outputs = model(inputs)
            features.append(outputs.cpu().numpy())
            labels.append(lbls.cpu().numpy())

            # Retrieve the image paths based on the subset indices
            image_paths.extend([base_dataset.samples[i][0] for i in subset_indices])

    features = np.vstack(features)
    labels = np.hstack(labels)

    return features, labels, image_paths


def find_prototypical_images_paths(prototypes, class_names, class_features, classified_labels, val_features, val_image_paths):
 # Prepare list to store results
        results = []

        # Step 3: Find closest image to the prototype for each class based on the KNN classification
        for selected_class in class_names:
            prototype = prototypes[selected_class]

            # Filter validation features and paths classified as the selected class
            class_indices = [i for i, label in enumerate(classified_labels) if label == selected_class]
            
            if len(class_indices) == 0:
                print(f"Warning: No samples classified as {selected_class}")
                continue

            class_features = val_features[class_indices]
            class_image_paths = [val_image_paths[i] for i in class_indices]  # Assuming val_image_paths contains paths to validation set images

            # Find the closest image to the prototype
            distances = np.linalg.norm(class_features - prototype, axis=1)
            closest_image_path = np.argmin(distances)
            print(f"The closest image to the prototype of class '{selected_class}' is: {closest_image_path}")

            # Append the result for this class
            results.append([selected_class, closest_image_path])

        # Step 4: Save the results to a CSV file
        with open('mushroom_prototypes.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Class", "Closest Image Path"])  # Header
            writer.writerows(results)  # Write all class/image path pairs

        print(f"Results saved to mushroom_prototypes.csv")
