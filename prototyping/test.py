import os
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from autoencoder import MushroomAutoencoder

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

autoencoder_path = "mushroom_autoencoder_2.pth"
input_dir = "reduced_dataset"
output_dir = "reduced_dataset_output"
batch_size = 64

transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize to 256x256
    transforms.ToTensor(),          # Convert to tensor (normalizes to [0, 1])
])

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Dataset and DataLoader
dataset = datasets.ImageFolder(root=input_dir, transform=transform)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


def load_autoencoder(model_path, input_channels=3, latent_dim=64):
    model = MushroomAutoencoder(input_channels=input_channels, latent_dim=latent_dim).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=torch.device(DEVICE)))
    model.eval()
    return model


model = load_autoencoder(autoencoder_path)
model.eval()  # Set model to evaluation mode

with torch.no_grad():
    for i, (batch, labels) in enumerate(train_loader):  # Only unpack batch and labels
        batch = batch.to(DEVICE)
        recon_batch, _, _ = model(batch)  # Unpack only the necessary outputs

        recon_batch = recon_batch.view(-1, 3, 256, 256).cpu()

        # Save the reconstructed images
        for j in range(recon_batch.size(0)):
            recon_img = transforms.ToPILImage()(recon_batch[j])
            img_name = f"reconstructed_{i * batch_size + j}.png"  # Create a unique filename
            recon_img.save(os.path.join(output_dir, img_name))

print(f"Reconstructed images saved to '{output_dir}'.")
