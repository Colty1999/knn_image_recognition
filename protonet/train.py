import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from autoencoder import VAEIdsia  # Import the new VAE model
from helpers import loss_function  # Ensure this computes both reconstruction and KL-divergence losses
from prototypes_loader import MushroomDataset
from torchvision import transforms

# Set device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
input_channels = 3
input_size = 224  # Set based on your transformations
latent_dim = 300
cnn_channels = [100, 150, 250]  # CNN layer channels as per the model definition
stn_params = ([32, 64, 128], [64, 128, 256], [128, 256, 512])  # Sample parameters for STNs

learning_rate = 1e-4
batch_size = 32
num_epochs = 1000
input_dir = "reduced_dataset"
csv_path = "mushroom_prototypes.csv"
initial_model_path = ""
previous_epoch = 0

# Load dataset
dataset = MushroomDataset(root_dir=input_dir, csv_file=csv_path)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize model and optimizer
model = VAEIdsia(nc=input_channels, input_size=input_size, latent_variable_size=latent_dim, cnn_chn=cnn_channels,
                 param1=stn_params[0], param2=stn_params[1], param3=stn_params[2]).to(DEVICE)

# Load model checkpoint if resuming training
if os.path.exists(initial_model_path):
    model.load_state_dict(torch.load(initial_model_path))
    print(f"Resuming training from {initial_model_path} at epoch {previous_epoch}.")
else:
    print("No checkpoint found; starting training from scratch.")

optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(previous_epoch, previous_epoch + num_epochs):
    model.train()
    running_loss = 0.0
    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch + 1}/{previous_epoch + num_epochs}")

    for i, (inputs, prototypes, labels) in progress_bar:
        inputs, prototypes = inputs.to(DEVICE), prototypes.to(DEVICE)

        optimizer.zero_grad()

        # Forward pass
        reconstructed, mu, logvar, xstn = model(inputs)

        # Compute loss (reconstruction + KL divergence)
        loss = loss_function(reconstructed, prototypes, mu, logvar)

        # Backpropagation
        loss.backward()
        optimizer.step()

        # Logging
        running_loss += loss.item()
        progress_bar.set_postfix(loss=running_loss / (i + 1))

    print(f"Epoch [{epoch + 1}/{previous_epoch + num_epochs}], Loss: {running_loss / len(train_loader):.4f}")

    # Save the model checkpoint and reconstructed outputs every 100 epochs
    if (epoch + 1) % 100 == 0:
        # Save the model
        model_save_path = f"mushroom_protonet_{epoch + 1}_epoch.pth"
        torch.save(model.state_dict(), model_save_path)
        print(f"Model saved to {model_save_path}")

        # Save reconstructed images for inspection
        output_dir = f"protonet_reduced_dataset_output_{epoch + 1}_epoch"
        os.makedirs(output_dir, exist_ok=True)
        model.eval()
        with torch.no_grad():
            for i, (batch, _, labels) in enumerate(train_loader):
                batch = batch.to(DEVICE)
                recon_batch, _, _, _ = model(batch)
                recon_batch = recon_batch.view(-1, 3, input_size, input_size).cpu()

                # Save reconstructed images into subfolders by class
                for j in range(recon_batch.size(0)):
                    label = labels[j]  # Assuming labels[j] gives the class name
                    class_output_dir = os.path.join(output_dir, label)
                    os.makedirs(class_output_dir, exist_ok=True)  # Create subfolder for the class if it doesn't exist

                    # Save the reconstructed image in the appropriate class subfolder
                    recon_img = transforms.ToPILImage()(recon_batch[j])
                    img_name = f"reconstructed_{i * batch_size + j}.png"
                    recon_img.save(os.path.join(class_output_dir, img_name))

        model.train()  # Switch back to training mode after saving outputs
