import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from torch.utils.data import DataLoader
from autoencoder import MushroomAutoencoder
from prototypes_loader import MushroomDataset

# Parameters
batch_size = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_dir = "reduced_dataset"
csv_path = "mushroom_prototypes.csv"
initial_model_path = "mushroom_autoencoder.pth"  # Path to the saved model

# Timestamp for the filename
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
save_path = f"tsne_latent_representations_{timestamp}.png"

# Initialize dataset and dataloader
dataset = MushroomDataset(root_dir=input_dir, csv_file=csv_path)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

# Load the trained model
model = MushroomAutoencoder(input_channels=3, latent_dim=256, image_size=256).to(DEVICE)
model.load_state_dict(torch.load(initial_model_path))
model.eval()

# Collect latent representations and corresponding labels
latent_representations = []
labels = []

with torch.no_grad():
    for inputs, _, class_names in dataloader:
        inputs = inputs.to(DEVICE)

        # Get latent representation
        mu, _ = model.encode(inputs)
        latent_vectors = mu.cpu().numpy()

        latent_representations.append(latent_vectors)
        labels.extend(class_names)  # Use class names directly as labels

# Flatten the list of latent representations
latent_representations = np.concatenate(latent_representations, axis=0)

# Use t-SNE to reduce to 2D
tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, random_state=0)
latent_2d = tsne.fit_transform(latent_representations)

# Plot the t-SNE results
plt.figure(figsize=(12, 10))
unique_labels = list(set(labels))
colors = plt.cm.jet(np.linspace(0, 1, len(unique_labels)))

for i, label in enumerate(unique_labels):
    indices = [j for j, lbl in enumerate(labels) if lbl == label]
    plt.scatter(latent_2d[indices, 0], latent_2d[indices, 1], label=label, color=colors[i], alpha=0.6, s=10)

plt.legend(markerscale=2, bbox_to_anchor=(1.05, 1), loc='upper left')
plt.title("t-SNE Visualization of Latent Space")
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.tight_layout()
plt.savefig(save_path, dpi=300)
plt.show()

print(f"t-SNE plot saved to {save_path}")
