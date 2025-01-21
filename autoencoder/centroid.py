import time
import torch
from torch.utils.data import DataLoader
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from autoencoder import MushroomAutoencoder
from prototypes_loader import MushroomDataset


def compute_centroids(train_loader, model):
    model.eval()
    DEVICE = next(model.parameters()).device

    centroids = {}
    class_counts = {}

    with torch.no_grad():
        for inputs, _, labels in train_loader:
            inputs = inputs.to(DEVICE)
            mu, _ = model.encode(inputs)
            for latent, label in zip(mu, labels):
                # Zakładamy, że label to string - bez `.item()`
                if label not in centroids:
                    centroids[label] = latent.cpu().numpy()
                    class_counts[label] = 1
                else:
                    centroids[label] += latent.cpu().numpy()
                    class_counts[label] += 1

    for label in centroids:
        centroids[label] /= class_counts[label]

    return centroids


def classify_using_centroids(test_loader, centroids, model):
    model.eval()
    DEVICE = next(model.parameters()).device
    correct = 0
    total = 0

    start_score = time.time()
    with torch.no_grad():
        for inputs, _, labels in test_loader:
            inputs = inputs.to(DEVICE)
            mu, _ = model.encode(inputs)
            for latent, label in zip(mu, labels):
                distances = {cls: np.linalg.norm(latent.cpu().numpy() - centroid) for cls, centroid in centroids.items()}
                predicted = min(distances, key=distances.get)
                if predicted == label:  # Porównanie stringów
                    correct += 1
                total += 1
    end_score = time.time()

    accuracy = correct / total
    print(f"Centroid Classifier Accuracy: {accuracy * 100:.2f}%")
    print(f"Time to score Centroid Classifier: {end_score - start_score:.4f} seconds")
    return accuracy


# Funkcja do obliczania centroidów na podstawie zbioru treningowego
def compute_centroids_with_prototypes(train_loader, model):
    model.eval()
    DEVICE = next(model.parameters()).device

    centroids = {}
    class_counts = {}

    with torch.no_grad():
        for inputs, prototypes, labels in train_loader:
            if inputs is None or prototypes is None or labels is None:
                print("Error: Missing data in train_loader batch!")
                continue

            inputs, prototypes = inputs.to(DEVICE), prototypes.to(DEVICE)
            mu, _ = model.encode(inputs)
            mu_proto, _ = model.encode(prototypes)
            combined_mu = (mu + mu_proto) / 2  # Uśrednienie cech

            for latent, label in zip(combined_mu, labels):
                if label not in centroids:
                    centroids[label] = latent.cpu().numpy()
                    class_counts[label] = 1
                else:
                    centroids[label] += latent.cpu().numpy()
                    class_counts[label] += 1

    if not centroids:
        print("Error: No centroids were computed. Check your data and model.")
        return None

    for label in centroids:
        centroids[label] /= class_counts[label]

    return centroids

# Funkcja do obliczania dokładności na podstawie centroidów
def classify_using_centroids_with_prototypes(test_loader, centroids, model):
    if centroids is None:
        print("Error: Centroids are None. Classification cannot proceed.")
        return None

    model.eval()
    DEVICE = next(model.parameters()).device
    correct = 0
    total = 0

    start_score = time.time()
    with torch.no_grad():
        for inputs, _, labels in test_loader:
            inputs = inputs.to(DEVICE)
            mu, _ = model.encode(inputs)
            for latent, label in zip(mu, labels):
                distances = {cls: np.linalg.norm(latent.cpu().numpy() - centroid) for cls, centroid in centroids.items()}
                predicted = min(distances, key=distances.get)
                if predicted == label:  # Porównanie stringów
                    correct += 1
                total += 1
    end_score = time.time()

    accuracy = correct / total
    print(f"Centroid with Prototypes Classifier Accuracy: {accuracy * 100:.2f}%")
    print(f"Time to score Centroid with Prototypes: {end_score - start_score:.4f} seconds")
    return accuracy


# Parametry
input_dir_train = "reduced_dataset"  # Ścieżka do zbioru treningowego
input_dir_test = "dataset"  # Ścieżka do zbioru testowego
csv_path = "mushroom_prototypes.csv"
model_path = "mushroom_autoencoder.pth"  # Ścieżka do modelu
batch_size = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Ładowanie danych
train_dataset = MushroomDataset(root_dir=input_dir_train, csv_file=csv_path)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = MushroomDataset(root_dir=input_dir_test, csv_file=csv_path)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Inicjalizacja modelu
model = MushroomAutoencoder(input_channels=3, latent_dim=256, image_size=256).to(DEVICE)
model.load_state_dict(torch.load(model_path))
model.eval()

# --- Klasyczny Centroid ---
print("=== Klasyczny Centroid ===")
centroids_classic = compute_centroids(train_loader, model)
accuracy_classic = classify_using_centroids(test_loader, centroids_classic, model)

# --- Centroid z Prototypami ---
print("\n=== Centroid z Prototypami ===")
centroids_prototyping = compute_centroids_with_prototypes(train_loader, model)
accuracy_prototyping = classify_using_centroids_with_prototypes(test_loader, centroids_prototyping, model)

# Podsumowanie wyników
print("\n--- Podsumowanie ---")
print(f"Accuracy (Klasyczny Centroid): {accuracy_classic * 100:.2f}%")
print(f"Accuracy (Centroid z Prototypami): {accuracy_prototyping * 100:.2f}%")
