import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from autoencoder import MushroomAutoencoder
from prototypes_loader import MushroomDataset

# Set device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Parameters
input_channels = 3
latent_dim = 256
image_size = 256
batch_size = 32
input_dir = "dataset"  # Directory containing images for testing
output_dir = "dataset_output"  # Directory to save reconstructed images
csv_path = "mushroom_prototypes.csv"  # Path to the CSV file
initial_model_path = "mushroom_autoencoder.pth"  # Path to the saved model

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Load dataset with the CSV file for labels
test_dataset = MushroomDataset(root_dir=input_dir, csv_file=csv_path)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Load the trained model
model = MushroomAutoencoder(input_channels=input_channels, latent_dim=256, image_size=image_size).to(DEVICE)
model.load_state_dict(torch.load(initial_model_path))
model.eval()  # Set the model to evaluation mode

# Transform to save images
to_pil_image = transforms.ToPILImage()

# Inference and saving reconstructed images
with torch.no_grad():
    for i, (inputs, _, labels) in tqdm(enumerate(test_loader), total=len(test_loader), desc="Reconstructing Images"):
        inputs = inputs.to(DEVICE)

        # Reconstruct images using the model
        reconstructed, _, _ = model(inputs)
        reconstructed = reconstructed.cpu()  # Move to CPU for saving

        # Save each reconstructed image in the appropriate subfolder
        for j in range(reconstructed.size(0)):
            label = labels[j]  # Assuming labels[j] gives the class name
            class_output_dir = os.path.join(output_dir, label)
            os.makedirs(class_output_dir, exist_ok=True)  # Create subfolder for the class if it doesn't exist

            # Save image in the class subfolder
            recon_img = to_pil_image(reconstructed[j])
            img_name = f"reconstructed_{i * batch_size + j}.png"
            recon_img.save(os.path.join(class_output_dir, img_name))

print(f"Reconstructed images have been saved to {output_dir}")
