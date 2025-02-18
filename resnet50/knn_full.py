import os
import torch
import numpy as np
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.neighbors import KNeighborsClassifier
import time
import torch.nn as nn

# Load trained model
def load_trained_model(model_path, num_classes, device):
    model = models.resnet50(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = nn.Sequential(*list(model.children())[:-1]) 
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


# Train and evaluate KNN
def train_and_evaluate_knn(train_features, train_labels, test_features, test_labels, n_neighbors=5):
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)

    # Fit KNN
    start_time_fit = time.time()
    knn.fit(train_features, train_labels)
    end_time_fit = time.time()

    # Score KNN
    start_time_score = time.time()
    accuracy = knn.score(test_features, test_labels)
    end_time_score = time.time()

    print(f"KNN Classifier Accuracy: {accuracy * 100:.2f}%")
    print(f"Time taken to fit the model: {end_time_fit - start_time_fit:.4f} seconds")
    print(f"Time taken to score the model: {end_time_score - start_time_score:.4f} seconds")

    return knn


def calculate_knn_with_prototypes(train_features, train_labels, test_features, test_labels):

    # Obliczanie prototypów jako średnich dla każdej klasy
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

    # Dopasowanie KNN przy użyciu prototypów
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(prototypes, prototype_labels)
    end_time_fit = time.time()

    # Ocena modelu KNN z prototypami
    start_time_score = time.time()
    accuracy = knn.score(test_features, test_labels)
    end_time_score = time.time()

    print(f"KNN Classifier Accuracy using prototypes: {accuracy * 100:.2f}%")
    print(f"Time taken to fit the model: {end_time_fit - start_time_fit:.4f} seconds")
    print(f"Time taken to score the model: {end_time_score - start_time_score:.4f} seconds")
    return accuracy


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = 'Resnet_model.pth'
    
    # Define dataset directories
    train_dir = "reduced_dataset"
    test_dir = "dataset"
    num_classes = len(os.listdir(train_dir))  # Assumes each subfolder is a class

    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    # Load datasets
    train_dataset = datasets.ImageFolder(train_dir, transform=transform)
    test_dataset = datasets.ImageFolder(test_dir, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Load trained model
    model = load_trained_model(model_path, num_classes, device)

    # Extract features
    print("Extracting features for training set...")
    train_features, train_labels = extract_features(model, train_loader, device)
    print("Extracting features for testing set...")
    test_features, test_labels = extract_features(model, test_loader, device)

    # Train and evaluate KNN
    print("Training and evaluating KNN...")
    train_and_evaluate_knn(train_features, train_labels, test_features, test_labels, n_neighbors=3)

    print("Calculating KNN using prototypes...")
    calculate_knn_with_prototypes(train_features, train_labels, test_features, test_labels)

if __name__ == "__main__":
    main()
