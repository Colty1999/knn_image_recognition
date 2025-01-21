import torch
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from torch.utils.data import DataLoader
from autoencoder import MushroomAutoencoder
from prototypes_loader import MushroomDataset
from collections import defaultdict

# Parameters
batch_size = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_dir = "dataset"
csv_path = "mushroom_prototypes.csv"
initial_model_path = "mushroom_autoencoder_2000_epoch.pth"  # Path to the saved model

# Timestamp for the filename
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

# Initialize dataset and dataloader
dataset = MushroomDataset(root_dir=input_dir, csv_file=csv_path)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

# Load the trained model
model = MushroomAutoencoder(input_channels=3, latent_dim=256, image_size=256).to(DEVICE)
model.load_state_dict(torch.load(initial_model_path))
model.eval()

# Collect latent representations and corresponding labels
class_latent_representations = defaultdict(list)

with torch.no_grad():
    for inputs, _, class_names in dataloader:
        inputs = inputs.to(DEVICE)

        # Get latent representation
        mu, _ = model.encode(inputs)
        latent_vectors = mu.cpu().numpy()

        for latent_vector, class_name in zip(latent_vectors, class_names):
            class_latent_representations[class_name].append(latent_vector)

# Average the latent representations for each class
prototypes = []
labels = []

for class_name, latent_vectors in class_latent_representations.items():
    prototype = np.mean(latent_vectors, axis=0)  # Compute the average
    prototypes.append(prototype)
    labels.append(class_name)

prototypes = np.array(prototypes)

# Apply PCA for noise reduction
pca = PCA(n_components=50)
prototypes_pca = pca.fit_transform(prototypes)

# Apply t-SNE to class prototypes
tsne = TSNE(n_components=2, perplexity=10, learning_rate=100, random_state=0)
prototypes_2d = tsne.fit_transform(prototypes_pca)

# Plot the t-SNE results
plt.figure(figsize=(12, 10))
colors = plt.cm.jet(np.linspace(0, 1, len(labels)))

for i, label in enumerate(labels):
    plt.scatter(
        prototypes_2d[i, 0],
        prototypes_2d[i, 1],
        label=label,
        color=colors[i],
        alpha=0.8,
        s=100,
    )

plt.legend(markerscale=2, bbox_to_anchor=(1.05, 1), loc="upper left")
plt.title("t-SNE Visualization of Class Prototypes")
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.tight_layout()
save_path = f"prototypes_tsne_class_prototypes_{timestamp}.png"
plt.savefig(save_path, dpi=300)
plt.show()

print(f"t-SNE plot of class prototypes saved to {save_path}")
