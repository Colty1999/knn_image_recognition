import torch
from torch.utils.data import DataLoader
from sklearn.neighbors import KNeighborsClassifier
from collections import defaultdict
import numpy as np
from autoencoder import MushroomAutoencoder
from prototypes_loader import MushroomDataset

# Parameters
input_dir = "reduced_dataset"
csv_path = "mushroom_prototypes.csv"
initial_model_path = "mushroom_autoencoder.pth"  # Path to the saved model
batch_size = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize dataset and dataloaders
train_dataset = MushroomDataset(root_dir=input_dir, csv_file=csv_path)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

test_dataset = MushroomDataset(root_dir=input_dir, csv_file=csv_path)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Load the trained model
model = MushroomAutoencoder(input_channels=3, latent_dim=256, image_size=256).to(DEVICE)
model.load_state_dict(torch.load(initial_model_path))
model.eval()

# Step 1: Extract latent representations and compute class prototypes
class_prototypes = defaultdict(list)

with torch.no_grad():
    for inputs, _, labels in train_loader:
        inputs = inputs.to(DEVICE)

        # Get latent representations from the encoder
        mu, _ = model.encode(inputs)
        latent_vectors = mu.cpu().numpy()

        # Group latent vectors by class label (string-based class names)
        for vec, label in zip(latent_vectors, labels):
            class_prototypes[label].append(vec)

# Step 2: Compute the mean vector for each class to create prototypes
class_prototypes = {label: np.mean(vectors, axis=0) for label, vectors in class_prototypes.items()}

# Prepare the data for KNN
X_train = np.array(list(class_prototypes.values()))  # Latent space prototypes for each class
y_train = np.array(list(class_prototypes.keys()))    # Corresponding class names (strings)

# Step 3: Train the KNN classifier on the class prototypes
knn = KNeighborsClassifier(n_neighbors=3)  # Adjust the number of neighbors as needed
knn.fit(X_train, y_train)

# Step 4: Evaluate KNN Classifier on the Test Set
correct = 0
total = 0

with torch.no_grad():
    for inputs, _, labels in test_loader:
        inputs = inputs.to(DEVICE)

        # Get latent representation of test images
        mu, _ = model.encode(inputs)
        latent_vectors = mu.cpu().numpy()

        # Predict using KNN based on the latent representations
        predictions = knn.predict(latent_vectors)
        
        # Count correct predictions
        correct += (predictions == np.array(labels)).sum()  # Compare strings directly
        total += len(labels)

accuracy = correct / total * 100
print(f"KNN Classifier Accuracy on Test Set: {accuracy:.2f}%")
