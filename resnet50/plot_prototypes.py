import os
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from datetime import datetime
import numpy as np
from helpers import load_data, extract_features
from train import initialize_model

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data
data_dir = "reduced_dataset"
batch_size = int(os.getenv('BATCH_SIZE'))
validation_split = float(os.getenv('VALIDATION_SPLIT'))

dataloaders, dataset_sizes, class_names = load_data(data_dir, batch_size, validation_split)

# Load pre-trained model
num_classes = len(class_names)
model = initialize_model(num_classes=num_classes)
model_path = os.getenv('MODEL_PATH')
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)

# Extract features from validation dataset
val_dataloader = dataloaders['val']
features, labels, _ = extract_features(model, val_dataloader)

# Compute mean feature vector for each class
class_prototypes = {}
for i, class_name in enumerate(class_names):
    class_indices = [index for index, label in enumerate(labels) if label == i]
    if len(class_indices) > 0:
        class_features = features[class_indices]
        class_prototypes[class_name] = np.mean(class_features, axis=0)
    else:
        print(f"Warning: No samples found for class {class_name}")

# Prepare the prototypes for t-SNE
prototype_features = np.array(list(class_prototypes.values()))
prototype_labels = list(class_prototypes.keys())

# Apply t-SNE
tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=300)
reduced_prototypes = tsne.fit_transform(prototype_features)

# Plot t-SNE results for prototypes
plt.figure(figsize=(10, 8))
for i, (label, coord) in enumerate(zip(prototype_labels, reduced_prototypes)):
    plt.scatter(coord[0], coord[1], label=label, s=100)  # Larger marker for prototypes

plt.title("t-SNE Visualization of Class Prototypes")
plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")
plt.legend(loc='best', bbox_to_anchor=(1.05, 1), title="Classes")
plt.tight_layout()

# Add timestamp to file name
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = f"tsne_prototypes_{timestamp}.png"
plt.savefig(output_file)
plt.show()

print(f"t-SNE visualization saved to {output_file}")
