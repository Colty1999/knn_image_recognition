import os
from torchvision import transforms
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from autoencoder import MushroomAutoencoder
from helpers import loss_function
from prototypes_loader import MushroomDataset  

# Set device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
input_channels = 3
latent_dim = 256
learning_rate = 1e-6
batch_size = 32
num_epochs = 500
image_size = 256
model_save_path = "mushroom_autoencoder.pth"
input_dir = "reduced_dataset"
output_dir = "reduced_dataset_output"
csv_path = "mushroom_prototypes.csv"

# Load dataset
dataset = MushroomDataset(root_dir=input_dir, csv_file=csv_path)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize model, optimizer
model = MushroomAutoencoder(input_channels=input_channels, latent_dim=latent_dim, image_size=image_size).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
model.train()
for epoch in range(num_epochs):
    running_loss = 0.0
    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch + 1}/{num_epochs}")

    for i, (inputs, prototypes, _) in progress_bar:
        inputs = inputs.to(DEVICE)
        prototypes = prototypes.to(DEVICE)

        optimizer.zero_grad()

        # Forward pass
        reconstructed, mu, log_var = model(inputs)

        # Compute loss
        loss = loss_function(reconstructed, prototypes, mu, log_var)

        # Backpropagation
        loss.backward()

        # Gradient clipping to avoid exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # Log
        running_loss += loss.item()
        progress_bar.set_postfix(loss=running_loss / (i + 1))

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}")

# Save the model
torch.save(model.state_dict(), model_save_path)

# Inference (Reconstructing and saving images)
model.eval()
os.makedirs(output_dir, exist_ok=True)
with torch.no_grad():
    for i, (batch, _, _) in enumerate(train_loader):
        batch = batch.to(DEVICE)
        recon_batch, _, _ = model(batch)
        recon_batch = recon_batch.view(-1, 3, image_size, image_size).cpu()

        # Save reconstructed images
        for j in range(recon_batch.size(0)):
            recon_img = transforms.ToPILImage()(recon_batch[j])
            img_name = f"reconstructed_{i * batch_size + j}.png"
            recon_img.save(os.path.join(output_dir, img_name))
