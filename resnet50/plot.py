import os
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from datetime import datetime
from helpers import load_data, extract_features
from train import initialize_model

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data
data_dir = "dataset"
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

# Apply t-SNE
tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=300)
reduced_features = tsne.fit_transform(features)

# Plot t-SNE results
plt.figure(figsize=(10, 8))
for i, class_name in enumerate(class_names):
    class_indices = [index for index, label in enumerate(labels) if label == i]
    plt.scatter(reduced_features[class_indices, 0], reduced_features[class_indices, 1], label=class_name, alpha=0.6)

plt.legend(loc='best', bbox_to_anchor=(1.05, 1), title="Classes")
plt.title("t-SNE Visualization of Class Features")
plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")
plt.tight_layout()

# Add timestamp to file name
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = f"tsne_visualization_{timestamp}.png"
plt.savefig(output_file)
plt.show()

print(f"t-SNE visualization saved to {output_file}")
