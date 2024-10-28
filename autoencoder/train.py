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
num_epochs = 3000  # Number of epochs to train in this session
image_size = 256
input_dir = "reduced_dataset"
csv_path = "mushroom_prototypes.csv"
initial_model_path = "mushroom_autoencoder.pth"  # Path to the saved model

# Variable to track the previous epoch if resuming training
previous_epoch = 2000  # Set this to the epoch number where you want to resume training

# Load dataset
dataset = MushroomDataset(root_dir=input_dir, csv_file=csv_path)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize model, optimizer
model = MushroomAutoencoder(input_channels=input_channels, latent_dim=latent_dim, image_size=image_size).to(DEVICE)

# Load the previously saved model's state_dict if resuming training
if os.path.exists(initial_model_path):
    model.load_state_dict(torch.load(initial_model_path))
    print(f"Loaded model from {initial_model_path}, resuming from epoch {previous_epoch}.")
else:
    print(f"Model file {initial_model_path} not found. Starting training from scratch.")

optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop for additional epochs
model.train()
for epoch in range(previous_epoch, previous_epoch + num_epochs):
    running_loss = 0.0
    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch + 1}/{previous_epoch + num_epochs}")

    for i, (inputs, prototypes, labels) in progress_bar:  # Assuming labels are the class names
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

    print(f"Epoch [{epoch + 1}/{previous_epoch + num_epochs}], Loss: {running_loss / len(train_loader):.4f}")

    # Save the model and reconstructed outputs every 100 epochs
    if (epoch + 1) % 100 == 0:
        # Save the model
        model_save_path = f"mushroom_autoencoder_{epoch + 1}_epoch.pth"
        torch.save(model.state_dict(), model_save_path)
        print(f"Model saved to {model_save_path}")

        # Inference (Reconstructing and saving images)
        output_dir = f"reduced_dataset_output_{epoch + 1}_epoch"
        os.makedirs(output_dir, exist_ok=True)
        model.eval()
        with torch.no_grad():
            for i, (batch, _, labels) in enumerate(train_loader):  # Including labels (class names)
                batch = batch.to(DEVICE)
                recon_batch, _, _ = model(batch)
                recon_batch = recon_batch.view(-1, 3, image_size, image_size).cpu()

                # Save reconstructed images into subfolders by class
                for j in range(recon_batch.size(0)):
                    label = labels[j]  # Assuming labels[j] gives the class name
                    class_output_dir = os.path.join(output_dir, label)
                    os.makedirs(class_output_dir, exist_ok=True)  # Create subfolder for the class if it doesn't exist

                    # Save image in the appropriate class subfolder
                    recon_img = transforms.ToPILImage()(recon_batch[j])
                    img_name = f"reconstructed_{i * batch_size + j}.png"
                    recon_img.save(os.path.join(class_output_dir, img_name))

        model.train()  # Switch back to training mode after saving outputs
