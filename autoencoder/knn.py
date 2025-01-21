import time
import torch
from torch.utils.data import DataLoader
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from autoencoder import MushroomAutoencoder
from prototypes_loader import MushroomDataset

def train_and_evaluate_knn(train_loader, val_loader, model, n_neighbors=5):
    """
    Trains and evaluates a KNN classifier on the latent space features extracted from a model.
    
    Args:
        train_loader (DataLoader): DataLoader for the training set.
        val_loader (DataLoader): DataLoader for the validation/test set.
        model (nn.Module): The trained model with an encoder.
        n_neighbors (int): Number of neighbors for KNN.
        
    Returns:
        knn (KNeighborsClassifier): The trained KNN classifier.
    """
    model.eval()
    DEVICE = next(model.parameters()).device  # Get model's device

    # Step 1: Extract latent features and labels from train_loader (dataset_reduced)
    train_features, train_labels = [], []
    with torch.no_grad():
        for inputs, _, labels in train_loader:
            inputs = inputs.to(DEVICE)
            mu, _ = model.encode(inputs)  # Extract latent representations
            train_features.append(mu.cpu().numpy())
            train_labels.extend(labels)
    train_features = np.concatenate(train_features, axis=0)

    # Step 2: Extract latent features and labels from val_loader (dataset)
    val_features, val_labels = [], []
    with torch.no_grad():
        for inputs, _, labels in val_loader:
            inputs = inputs.to(DEVICE)
            mu, _ = model.encode(inputs)  # Extract latent representations
            val_features.append(mu.cpu().numpy())
            val_labels.extend(labels)
    val_features = np.concatenate(val_features, axis=0)

    # Step 3: Train KNN on the training set and evaluate on the validation set
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    
    start_time_fit = time.time()
    knn.fit(train_features, train_labels)  # Fit KNN on training set
    end_time_fit = time.time()

    start_time_score = time.time()
    accuracy = knn.score(val_features, val_labels)  # Predict and evaluate on the validation set
    end_time_score = time.time()

    print(f"KNN Classifier Accuracy: {accuracy * 100:.2f}%")
    print(f"Time taken to fit the model for k={n_neighbors}: {end_time_fit - start_time_fit:.4f} seconds")
    print(f"Time taken to score the model: {end_time_score - start_time_score:.4f} seconds")

    return knn

from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import time
import torch


def knn_with_prototypes(train_loader, val_loader, model, n_neighbors=5):
    model.eval()
    DEVICE = next(model.parameters()).device

    # Ekstrakcja cech i etykiet ze zbioru treningowego
    train_features, train_labels = [], []
    with torch.no_grad():
        for inputs, prototypes, labels in train_loader:
            inputs, prototypes = inputs.to(DEVICE), prototypes.to(DEVICE)
            mu, _ = model.encode(inputs)  # Cechy z obrazów
            mu_proto, _ = model.encode(prototypes)  # Cechy z prototypów
            combined_mu = (mu + mu_proto) / 2  # Uśrednienie
            train_features.append(combined_mu.cpu().numpy())
            train_labels.extend(labels)
    train_features = np.concatenate(train_features, axis=0)

    # Ekstrakcja cech i etykiet ze zbioru walidacyjnego
    val_features, val_labels = [], []
    with torch.no_grad():
        for inputs, prototypes, labels in val_loader:
            inputs, prototypes = inputs.to(DEVICE), prototypes.to(DEVICE)
            mu, _ = model.encode(inputs)
            mu_proto, _ = model.encode(prototypes)
            combined_mu = (mu + mu_proto) / 2
            val_features.append(combined_mu.cpu().numpy())
            val_labels.extend(labels)
    val_features = np.concatenate(val_features, axis=0)

    # Trenowanie i ocena KNN
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    start_fit = time.time()
    knn.fit(train_features, train_labels)
    end_fit = time.time()

    start_score = time.time()
    accuracy = knn.score(val_features, val_labels)
    end_score = time.time()

    print(f"KNN Accuracy with Prototypes: {accuracy * 100:.2f}%")
    print(f"Time to fit KNN for k={n_neighbors}: {end_fit - start_fit:.4f} seconds")
    print(f"Time to score KNN: {end_score - start_score:.4f} seconds")

    return knn


# Parameters
input_dir_train = "reduced_dataset"  # Directory for training
input_dir_test = "dataset"  # Directory for testing
csv_path = "mushroom_prototypes.csv"
initial_model_path = "mushroom_autoencoder.pth"  # Path to the saved model
batch_size = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create dataset and DataLoader for training (dataset_reduced)
train_dataset = MushroomDataset(root_dir=input_dir_train, csv_file=csv_path)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Create dataset and DataLoader for testing (dataset)
test_dataset = MushroomDataset(root_dir=input_dir_test, csv_file=csv_path)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Load the trained model
model = MushroomAutoencoder(input_channels=3, latent_dim=256, image_size=256).to(DEVICE)
model.load_state_dict(torch.load(initial_model_path))
model.eval()

# Train KNN on dataset_reduced and predict on dataset
# for i in range(0, 4):
train_and_evaluate_knn(train_loader, test_loader, model, n_neighbors=5)
# for i in range(0, 4):
#     train_and_evaluate_knn(train_loader, test_loader, model, n_neighbors=3)
# for i in range(0, 4):
knn_with_prototypes(train_loader, test_loader, model, n_neighbors=1)
