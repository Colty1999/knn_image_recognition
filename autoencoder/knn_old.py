import time
import torch
from torch.utils.data import DataLoader
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from collections import defaultdict
import numpy as np
from autoencoder import MushroomAutoencoder
from prototypes_loader import MushroomDataset

def train_and_evaluate_knn(train_loader, val_loader, model, n_neighbors=5, n_components=None):
    """
    Trains and evaluates a KNN classifier on the latent space features extracted from a model.
    
    Args:
        train_loader (DataLoader): DataLoader for the training set.
        val_loader (DataLoader): DataLoader for the validation set.
        model (nn.Module): The trained model with an encoder.
        n_neighbors (int): Number of neighbors for KNN.
        n_components (int): Number of PCA components, set to None to skip PCA.
        
    Returns:
        knn (KNeighborsClassifier): The trained KNN classifier.
    """
    model.eval()
    DEVICE = next(model.parameters()).device  # Get model's device

    # Step 1: Extract latent features and labels from train_loader
    train_features, train_labels = [], []
    with torch.no_grad():
        for inputs, _, labels in train_loader:
            inputs = inputs.to(DEVICE)
            mu, _ = model.encode(inputs)  # Extract latent representations
            train_features.append(mu.cpu().numpy())
            train_labels.extend(labels)
    train_features = np.concatenate(train_features, axis=0)

    # Step 2: Extract latent features and labels from val_loader
    val_features, val_labels = [], []
    with torch.no_grad():
        for inputs, _, labels in val_loader:
            inputs = inputs.to(DEVICE)
            mu, _ = model.encode(inputs)  # Extract latent representations
            val_features.append(mu.cpu().numpy())
            val_labels.extend(labels)
    val_features = np.concatenate(val_features, axis=0)

    # Optional PCA step for dimensionality reduction
    if n_components is not None and n_components < train_features.shape[1]:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=n_components)
        train_features = pca.fit_transform(train_features)
        val_features = pca.transform(val_features)

    # Step 3: Train KNN and evaluate
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    
    start_time_fit = time.time()
    knn.fit(train_features, train_labels)
    end_time_fit = time.time()

    start_time_score = time.time()
    accuracy = knn.score(val_features, val_labels)
    end_time_score = time.time()

    print(f"KNN Classifier Accuracy: {accuracy * 100:.2f}%")
    print(f"Time taken to fit the model: {end_time_fit - start_time_fit:.4f} seconds")
    print(f"Time taken to score the model: {end_time_score - start_time_score:.4f} seconds")

    return knn

# Parameters
input_dir = "reduced_dataset"
csv_path = "mushroom_prototypes.csv"
initial_model_path = "mushroom_autoencoder.pth"  # Path to the saved model
batch_size = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Assuming MushroomDataset returns (image, prototype_image, class_name)
full_dataset = MushroomDataset(root_dir=input_dir, csv_file=csv_path)
train_indices, test_indices = train_test_split(
    range(len(full_dataset)),
    test_size=0.2,  # 20% for testing
    stratify=full_dataset.class_names  # Stratify by class if available
)

train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
val_dataset = torch.utils.data.Subset(full_dataset, test_indices)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Load the trained model
model = MushroomAutoencoder(input_channels=3, latent_dim=256, image_size=256).to(DEVICE)
model.load_state_dict(torch.load(initial_model_path))
model.eval()

train_and_evaluate_knn(train_loader, val_loader, model, n_neighbors=1, n_components=None)



# import torch
# from torch.utils.data import DataLoader
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.model_selection import train_test_split
# from collections import defaultdict
# import numpy as np
# from autoencoder import MushroomAutoencoder
# from prototypes_loader import MushroomDataset

# # Parameters
# input_dir = "reduced_dataset"
# csv_path = "mushroom_prototypes.csv"
# initial_model_path = "mushroom_autoencoder.pth"  # Path to the saved model
# batch_size = 32
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Assuming MushroomDataset returns (image, prototype_image, class_name)
# full_dataset = MushroomDataset(root_dir=input_dir, csv_file=csv_path)
# train_indices, test_indices = train_test_split(
#     range(len(full_dataset)),
#     test_size=0.2,  # 20% for testing
#     stratify=full_dataset.class_names  # Stratify by class if available
# )

# train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
# test_dataset = torch.utils.data.Subset(full_dataset, test_indices)

# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# # Load the trained model
# model = MushroomAutoencoder(input_channels=3, latent_dim=256, image_size=256).to(DEVICE)
# model.load_state_dict(torch.load(initial_model_path))
# model.eval()

# # Step 1: Extract latent representations and compute class prototypes
# class_prototypes = defaultdict(list)

# with torch.no_grad():
#     for inputs, _, labels in train_loader:
#         inputs = inputs.to(DEVICE)

#         # Get latent representations from the encoder
#         mu, _ = model.encode(inputs)
#         latent_vectors = mu.cpu().numpy()

#         # Group latent vectors by class label (string-based class names)
#         for vec, label in zip(latent_vectors, labels):
#             class_prototypes[label].append(vec)

# # Step 2: Compute the mean vector for each class to create prototypes
# class_prototypes = {label: np.mean(vectors, axis=0) for label, vectors in class_prototypes.items()}

# # Prepare the data for KNN
# X_train = np.array(list(class_prototypes.values()))  # Latent space prototypes for each class
# y_train = np.array(list(class_prototypes.keys()))    # Corresponding class names (strings)

# # Step 3: Train the KNN classifier on the class prototypes
# knn = KNeighborsClassifier(n_neighbors=3)  # Adjust the number of neighbors as needed
# knn.fit(X_train, y_train)

# # Step 4: Evaluate KNN Classifier on the Test Set
# correct = 0
# total = 0

# with torch.no_grad():
#     for inputs, _, labels in test_loader:
#         inputs = inputs.to(DEVICE)

#         # Get latent representation of test images
#         mu, _ = model.encode(inputs)
#         latent_vectors = mu.cpu().numpy()

#         # Predict using KNN based on the latent representations
#         predictions = knn.predict(latent_vectors)
        
#         # Count correct predictions
#         # print(predictions)
#         # print(labels)
#         correct += (predictions == np.array(labels)).sum()  # Compare strings directly
#         total += len(labels)

# accuracy = correct / total * 100
# print(f"KNN Classifier Accuracy on Test Set: {accuracy:.2f}%")
