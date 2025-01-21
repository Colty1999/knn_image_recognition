import os
import torch
import numpy as np
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import time
import torch.nn as nn

# Load trained model
def load_trained_model(model_path, num_classes, device):
    model = models.resnet50(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = nn.Sequential(*list(model.children())[:-1])  # Remove the final classification layer
    model = model.to(device)
    model.eval()
    return model

def extract_features(model, dataloader, device):
    features = []
    labels = []
    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            outputs = model(images)
            outputs = torch.flatten(outputs, start_dim=1)  # Flatten features
            features.append(outputs.cpu().numpy())
            labels.extend(targets.numpy())
    return np.vstack(features), np.array(labels)

def calculate_centroid_classifier(train_features, train_labels, test_features, test_labels):
    """
    Oblicza klasyfikację przy użyciu klasyfikatora centroidów.

    Args:
        train_features (numpy.ndarray): Wektory cech zbioru treningowego.
        train_labels (numpy.ndarray): Etykiety zbioru treningowego.
        test_features (numpy.ndarray): Wektory cech zbioru testowego.
        test_labels (numpy.ndarray): Etykiety zbioru testowego.

    Returns:
        float: Dokładność klasyfikacji na zbiorze testowym.
    """
    start_time_fit = time.time()
    unique_classes = np.unique(train_labels)
    centroids = []
    centroid_labels = []

    for cls in unique_classes:
        class_features = train_features[train_labels == cls]
        class_mean = np.mean(class_features, axis=0)
        centroids.append(class_mean)
        centroid_labels.append(cls)

    centroids = np.array(centroids)
    centroid_labels = np.array(centroid_labels)
    end_time_fit = time.time()

    correct = 0
    total = len(test_labels)
    start_time_score = time.time()

    for i, test_sample in enumerate(test_features):
        distances = np.linalg.norm(centroids - test_sample, axis=1)
        closest_idx = np.argmin(distances)
        predicted_label = centroid_labels[closest_idx]

        if predicted_label == test_labels[i]:
            correct += 1

    end_time_score = time.time()
    accuracy = correct / total

    print(f"Centroid Classifier Accuracy: {accuracy * 100:.2f}%")
    print(f"Time taken to compute centroids: {end_time_fit - start_time_fit:.4f} seconds")
    print(f"Time taken to classify using centroids: {end_time_score - start_time_score:.4f} seconds")
    return accuracy

def calculate_centroid_with_prototypes(train_features, train_labels, test_features, test_labels):
    """
    Oblicza klasyfikację centroidów przy użyciu prototypów jako dodatkowego kroku.

    Args:
        train_features (numpy.ndarray): Wektory cech zbioru treningowego.
        train_labels (numpy.ndarray): Etykiety zbioru treningowego.
        test_features (numpy.ndarray): Wektory cech zbioru testowego.
        test_labels (numpy.ndarray): Etykiety zbioru testowego.

    Returns:
        float: Dokładność klasyfikacji na zbiorze testowym.
    """
    start_time_fit = time.time()
    unique_classes = np.unique(train_labels)
    prototypes = []
    prototype_labels = []

    for cls in unique_classes:
        class_features = train_features[train_labels == cls]
        class_mean = np.mean(class_features, axis=0)
        prototypes.append(class_mean)
        prototype_labels.append(cls)

    prototypes = np.array(prototypes)
    prototype_labels = np.array(prototype_labels)

    centroids = []
    for i, cls in enumerate(unique_classes):
        prototype = prototypes[i]
        class_features = train_features[train_labels == cls]
        combined_mean = np.mean(np.vstack([class_features, prototype]), axis=0)
        centroids.append(combined_mean)

    centroids = np.array(centroids)
    end_time_fit = time.time()

    correct = 0
    total = len(test_labels)
    start_time_score = time.time()

    for i, test_sample in enumerate(test_features):
        distances = np.linalg.norm(centroids - test_sample, axis=1)
        closest_idx = np.argmin(distances)
        predicted_label = unique_classes[closest_idx]

        if predicted_label == test_labels[i]:
            correct += 1

    end_time_score = time.time()
    accuracy = correct / total

    print(f"Centroid with Prototypes Classifier Accuracy: {accuracy * 100:.2f}%")
    print(f"Time taken to compute centroids with prototypes: {end_time_fit - start_time_fit:.4f} seconds")
    print(f"Time taken to classify using centroids with prototypes: {end_time_score - start_time_score:.4f} seconds")
    return accuracy

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = 'Resnet_model.pth'

    train_dir = "reduced_dataset"
    test_dir = "dataset"
    num_classes = len(os.listdir(train_dir))

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    train_dataset = datasets.ImageFolder(train_dir, transform=transform)
    test_dataset = datasets.ImageFolder(test_dir, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = load_trained_model(model_path, num_classes, device)

    print("Extracting features for training set...")
    train_features, train_labels = extract_features(model, train_loader, device)
    print("Extracting features for testing set...")
    test_features, test_labels = extract_features(model, test_loader, device)

    print("Calculating Centroid Classifier...")
    calculate_centroid_classifier(train_features, train_labels, test_features, test_labels)

    print("Calculating Centroid with Prototypes Classifier...")
    calculate_centroid_with_prototypes(train_features, train_labels, test_features, test_labels)

if __name__ == "__main__":
    main()
